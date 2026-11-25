"""BatchTranscribe - Flet UI for faster-whisper batch transcription.

Run in dev:
    python main.py

Build a Windows .exe:
    build_windows.bat
"""
from __future__ import annotations

import json
import asyncio
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import flet as ft

from transcriber import BatchTranscriber, get_import_error, check_cuda, SUPPORTED_EXTS

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
APP_NAME = "BatchTranscribe"
APP_VERSION = "2.0.0"
MEDIA_FILTER = ";".join(f"*{ext}" for ext in SUPPORTED_EXTS)

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "turbo"]
LANGUAGES = [
    ("Auto-detect", None),
    ("English", "en"), ("Spanish", "es"), ("French", "fr"),
    ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
    ("Dutch", "nl"), ("Russian", "ru"), ("Japanese", "ja"),
    ("Korean", "ko"), ("Chinese", "zh"), ("Arabic", "ar"),
    ("Hindi", "hi"), ("Romanian", "ro"), ("Polish", "pl"),
    ("Swedish", "sv"), ("Turkish", "tr"), ("Vietnamese", "vi"),
    ("Thai", "th"),
]

# Color palette (dark by default, light available)
PALETTES = {
    "dark": {
        "bg_a": "#0A0E27", "bg_b": "#1A1F3A", "bg_c": "#0F1729",
        "surface": "#1E2444", "surface_alt": "#262C4D",
        "border": "#2D3461", "text_primary": "#F8FAFC",
        "text_secondary": "#94A3B8", "text_dim": "#64748B",
        "accent_a": "#6366F1", "accent_b": "#06B6D4",
        "success": "#10B981", "error": "#EF4444", "warning": "#F59E0B",
        "shadow": "#00000040",
    },
    "light": {
        "bg_a": "#FAFAFA", "bg_b": "#F0F4FF", "bg_c": "#E0E7FF",
        "surface": "#FFFFFF", "surface_alt": "#F8FAFC",
        "border": "#E2E8F0", "text_primary": "#0F172A",
        "text_secondary": "#475569", "text_dim": "#94A3B8",
        "accent_a": "#6366F1", "accent_b": "#06B6D4",
        "success": "#10B981", "error": "#EF4444", "warning": "#F59E0B",
        "shadow": "#0F172A20",
    },
}

CONFIG_PATH = Path.home() / ".batchtranscribe.json"


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


def fmt_size(n: int | None) -> str:
    if not n:
        return "-"
    units = ["B", "KB", "MB", "GB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.1f} {units[i]}"


# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
class App:
    def __init__(self, page: ft.Page):
        self.page = page
        self.cfg = load_config()
        self.theme_name = self.cfg.get("theme", "dark")
        if self.theme_name not in PALETTES:
            self.theme_name = "dark"
        self.palette = PALETTES[self.theme_name]

        # State
        self.files: list[dict] = []  # [{"path": str, "name": str, "size": int, "status": str}]
        self.file_rows: dict[str, ft.Container] = {}
        self.output_dir: str = ""
        self.queue: queue.Queue = queue.Queue()
        self.transcriber: BatchTranscriber | None = None
        self.worker: threading.Thread | None = None
        self.running = False

        # Detect CUDA up front (no subprocess, no shell)
        self.cuda_available = False
        self.cuda_message = "Checking transcription backend..."
        self.cuda_checked = False
        self.backend_ready = False

        # UI references
        self.file_list_col: ft.Column | None = None
        self.log_list: ft.Column | None = None
        self.status_label: ft.Text | None = None
        self.start_btn: ft.ElevatedButton | None = None
        self.cancel_btn: ft.OutlinedButton | None = None
        self.progress_bar: ft.ProgressBar | None = None
        self.drop_zone: ft.Container | None = None
        self.model_dd: ft.Dropdown | None = None
        self.device_dd: ft.Dropdown | None = None
        self.lang_dd: ft.Dropdown | None = None
        self.txt_cb: ft.Checkbox | None = None
        self.srt_cb: ft.Checkbox | None = None
        self.out_label: ft.Text | None = None
        self.loading_overlay: ft.Container | None = None
        self.loading_text: ft.Text | None = None

        # Build
        self.page.title = f"{APP_NAME} v{APP_VERSION}"
        self.page.padding = 0
        self.page.bgcolor = self.palette["bg_a"]
        self.page.window.width = 1100
        self.page.window.height = 760
        self.page.window.min_width = 900
        self.page.window.min_height = 620
        self.page.theme_mode = (
            ft.ThemeMode.DARK if self.theme_name == "dark" else ft.ThemeMode.LIGHT
        )
        font_dir = Path(__file__).parent / "assets" / "fonts"
        self.page.fonts = {name: str(font_dir / filename) for name, filename in
                           (("Inter", "Inter-Regular.ttf"), ("Inter-Bold", "Inter-Bold.ttf"))
                           if (font_dir / filename).is_file()}
        self.bold_font = "Inter-Bold" if "Inter-Bold" in self.page.fonts else None
        self.page.theme = ft.Theme(
            font_family="Inter" if "Inter" in self.page.fonts else None,
            color_scheme_seed=self.palette["accent_a"],
        )
        self.page.on_error = self._on_page_error

        # File picker service (Flet 0.85+)
        self.file_picker = ft.FilePicker()

        # Detect faster_whisper import error early so we can show a banner
        self.import_error = None

        self._build_ui()
        self._start_poll()

        # Entry animation
        self.page.update()
        self.page.run_task(self._initialize_backend)

    async def _initialize_backend(self):
        # Native imports and DLL verification must not block first paint.
        await asyncio.sleep(0.1)
        try:
            self.import_error = await asyncio.to_thread(get_import_error)
            if self.import_error is None:
                self.cuda_available, self.cuda_message = await asyncio.to_thread(check_cuda)
                self.backend_ready = True
        except Exception as exc:
            self.import_error = exc
        self.cuda_checked = True
        self.device_dd.options = [ft.DropdownOption(d) for d in (["cuda", "cpu"] if self.cuda_available else ["cpu"])]
        self.device_dd.value = "cuda" if self.cuda_available else "cpu"
        self.device_dd.update()
        self.backend_caption.value = "Whisper-powered batch transcription � " + ("CUDA accelerated" if self.cuda_available else "CPU mode")
        self.backend_caption.update()
        self.status_label.value = "Backend unavailable" if self.import_error else ("Ready � CUDA" if self.cuda_available else "Ready � CPU")
        self.status_label.update()
        self._update_start_enabled()
        if self.import_error:
            self._log("error", str(self.import_error))
            self._show_error_dialog("Backend import failed", str(self.import_error))
        else:
            self._log("info" if self.cuda_available else "warn", self.cuda_message)

    # ----- detection -----
    @staticmethod
    def _detect_cuda() -> bool:
        try:
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        except Exception:
            return False

    # ----- ui construction -----
    def _build_ui(self):
        p = self.palette

        # Outer stack: gradient background + main content + loading overlay
        gradient_bg = ft.Container(
            expand=True,
            gradient=ft.RadialGradient(
                center=ft.Alignment.TOP_LEFT,
                radius=1.4,
                colors=[p["bg_a"], p["bg_b"], p["bg_c"]],
                stops=[0.0, 0.6, 1.0],
            ),
        )

        main = self._build_main_content()
        loading = self._build_loading_overlay()
        self.loading_overlay = loading

        self.page.add(
            ft.Stack(
                controls=[gradient_bg, main, loading],
                expand=True,
                fit=ft.StackFit.EXPAND,
            )
        )

        # If the backend failed to import, surface it loudly
        if self.import_error:
            self._show_error_dialog(
                "Backend import failed",
                f"faster_whisper could not be loaded:\n\n{self.import_error}\n\n"
                "File selection remains available, but transcription requires a working backend. "
                "If Windows Security blocked a Python module, the package needs a trusted dependency build.",
            )

    def _build_main_content(self) -> ft.Container:
        p = self.palette
        return ft.Container(
            expand=True,
            padding=ft.Padding.symmetric(horizontal=32, vertical=20),
            content=ft.Column(
                expand=True,
                spacing=16,
                controls=[
                    self._build_topbar(),
                    ft.Row(
                        expand=True,
                        spacing=20,
                        controls=[
                            ft.Column(
                                expand=True,
                                spacing=16,
                                controls=[
                                    self._build_drop_zone(),
                                    self._build_file_list_card(),
                                ],
                            ),
                            ft.Column(
                                width=380,
                                spacing=16,
                                controls=[
                                    self._build_settings_card(),
                                    self._build_action_card(),
                                    self._build_log_card(),
                                ],
                            ),
                        ],
                    ),
                    self._build_status_bar(),
                ],
            ),
            opacity=1.0,
        )

    def _build_topbar(self) -> ft.Row:
        p = self.palette
        is_dark = self.theme_name == "dark"

        mic = ft.Container(
            width=44, height=44, border_radius=12,
            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment.BOTTOM_RIGHT,
                colors=[p["accent_a"], p["accent_b"]],
            ),
            content=ft.Icon(ft.Icons.MIC, color=ft.Colors.WHITE, size=24),
            shadow=ft.BoxShadow(blur_radius=20, color=p["accent_a"] + "60"),
        )

        self.backend_caption = ft.Text("Whisper-powered batch transcription � " + ("Checking backend" if not self.cuda_checked else ("CUDA accelerated" if self.cuda_available else "CPU mode")), size=12, color=p["text_secondary"])
        title_block = ft.Column(
            spacing=2,
            controls=[
                ft.Text(APP_NAME, size=24, weight=ft.FontWeight.BOLD,
                        color=p["text_primary"], font_family=self.bold_font),
                self.backend_caption,
            ],
        )

        def toggle_theme(_):
            self.theme_name = "light" if self.theme_name == "dark" else "dark"
            self.cfg["theme"] = self.theme_name
            save_config(self.cfg)
            self.palette = PALETTES[self.theme_name]
            self._rebuild_theme()

        theme_btn = ft.Container(
            width=44, height=44, border_radius=22,
            bgcolor=p["surface"], border=ft.Border.all(1, p["border"]),
            content=ft.Icon(
                ft.Icons.LIGHT_MODE if is_dark else ft.Icons.DARK_MODE,
                color=p["text_primary"], size=20,
            ),
            on_click=toggle_theme,
            tooltip="Toggle theme",
            ink=True,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )

        return ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Row(spacing=14, controls=[mic, title_block]),
                theme_btn,
            ],
        )

    def _build_drop_zone(self) -> ft.Container:
        p = self.palette

        def on_drag_will_accept(e):
            self.drop_zone.border = ft.Border.all(2, p["accent_a"])
            self.drop_zone.bgcolor = p["surface_alt"]
            self.drop_zone.scale = ft.Scale(1.02)
            self.drop_zone.update()

        def on_drag_leave(e):
            self.drop_zone.border = ft.Border.all(2, p["border"])
            self.drop_zone.bgcolor = p["surface"]
            self.drop_zone.scale = ft.Scale(1.0)
            self.drop_zone.update()

        def on_drag_accept(e):
            # Flet desktop drag payload: a string of file paths separated by \n
            # or a single path. Parse defensively.
            paths_raw = (e.data or "").strip()
            accepted: list[str] = []
            for line in paths_raw.replace("\r", "").split("\n"):
                # Sometimes includes a "file://" prefix on some platforms
                line = line.strip()
                if line.startswith("file://"):
                    line = line[len("file://"):]
                if line and os.path.exists(line):
                    accepted.append(line)
            if accepted:
                self._add_paths(accepted)
            on_drag_leave(e)

        self.drop_zone = ft.Container(
            height=160,
            border_radius=20,
            bgcolor=p["surface"],
            border=ft.Border.all(2, p["border"]),
            padding=ft.Padding.all(24),
            alignment=ft.Alignment.CENTER,
            ink=True,
            on_click=self._pick_files,
            animate=ft.Animation(250, ft.AnimationCurve.EASE_OUT),
            animate_scale=ft.Animation(250, ft.AnimationCurve.EASE_OUT),
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
                controls=[
                    ft.Container(
                        width=52, height=52, border_radius=26,
                        bgcolor=p["surface_alt"],
                        content=ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED,
                                        color=p["accent_a"], size=28),
                    ),
                    ft.Text("Drop audio/video here, or click to browse",
                            size=15, weight=ft.FontWeight.W_500,
                            color=p["text_primary"]),
                    ft.Text("MP3 � MP4 � WAV � FLAC � M4A � MKV � and more",
                            size=11, color=p["text_secondary"]),
                ],
            ),
        )

        # Wrap in DragTarget
        return ft.DragTarget(
            group="files",
            content=self.drop_zone,
            on_will_accept=on_drag_will_accept,
            on_accept=on_drag_accept,
            on_leave=on_drag_leave,
        )

    def _build_file_list_card(self) -> ft.Container:
        p = self.palette

        header = ft.Row(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Text("Files", size=14, weight=ft.FontWeight.W_600,
                        color=p["text_primary"]),
                ft.Row(
                    spacing=8,
                    controls=[
                        ft.OutlinedButton(
                            "Add folder", icon=ft.Icons.FOLDER_OPEN,
                            on_click=self._pick_input_folder,
                            style=ft.ButtonStyle(color=p["text_secondary"]),
                        ),
                        ft.TextButton(
                            "Clear", icon=ft.Icons.CLEAR_ALL,
                            on_click=self._clear_files,
                            style=ft.ButtonStyle(color=p["text_dim"]),
                        ),
                    ],
                ),
            ],
        )

        self.file_list_col = ft.Column(
            expand=True, spacing=6, scroll=ft.ScrollMode.AUTO,
        )

        empty_state = ft.Container(
            visible=not self.files,
            expand=True,
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=6,
                controls=[
                    ft.Icon(ft.Icons.INBOX_OUTLINED, size=36,
                            color=p["text_dim"]),
                    ft.Text("No files yet", color=p["text_dim"], size=13),
                ],
            ),
        )

        # Wrap in Stack so empty state shows when there are no files
        self.empty_state = empty_state
        self.file_stack = ft.Stack(
            expand=True,
            fit=ft.StackFit.EXPAND,
            controls=[self.file_list_col, empty_state],
        )

        return ft.Container(
            expand=True,
            padding=ft.Padding.all(20),
            border_radius=20,
            bgcolor=p["surface"],
            border=ft.Border.all(1, p["border"]),
            content=ft.Column(
                expand=True, spacing=12,
                controls=[header, self.file_stack],
            ),
        )

    def _build_settings_card(self) -> ft.Container:
        p = self.palette

        self.model_dd = ft.Dropdown(
            label="Model",
            value="medium",
            options=[ft.DropdownOption(m) for m in MODEL_SIZES],
            border_color=p["border"],
            text_style=ft.TextStyle(color=p["text_primary"]),
            label_style=ft.TextStyle(color=p["text_secondary"]),
            dense=True,
        )

        device_options = ["cuda", "cpu"] if self.cuda_available else ["cpu"]
        self.device_dd = ft.Dropdown(
            label="Device",
            value="cuda" if self.cuda_available else "cpu",
            options=[ft.DropdownOption(d) for d in device_options],
            border_color=p["border"],
            text_style=ft.TextStyle(color=p["text_primary"]),
            label_style=ft.TextStyle(color=p["text_secondary"]),
            dense=True,
        )

        self.lang_dd = ft.Dropdown(
            label="Language",
            value="Auto-detect",
            options=[ft.DropdownOption(name) for name, _ in LANGUAGES],
            border_color=p["border"],
            text_style=ft.TextStyle(color=p["text_primary"]),
            label_style=ft.TextStyle(color=p["text_secondary"]),
            dense=True,
        )

        self.txt_cb = ft.Checkbox(label=".txt", value=True,
                                   fill_color=p["accent_a"])
        self.srt_cb = ft.Checkbox(label=".srt", value=True,
                                   fill_color=p["accent_a"])
        self.diarize_cb = ft.Checkbox(label="Speaker labels", value=False, tooltip="Detect speakers locally; models download automatically if missing")
        self.fast_cb = ft.Checkbox(label="Fast decoding", value=False, tooltip="Greedy decoding; may reduce accuracy")

        return ft.Container(
            padding=ft.Padding.all(20),
            border_radius=20,
            bgcolor=p["surface"],
            border=ft.Border.all(1, p["border"]),
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Text("Settings", size=14, weight=ft.FontWeight.W_600,
                            color=p["text_primary"]),
                    ft.Row(spacing=12, controls=[self.model_dd, self.device_dd]),
                    self.lang_dd,
                    ft.Row(
                        spacing=8,
                        controls=[
                            ft.Text("Output:",
                                    color=p["text_secondary"], size=12),
                            self.txt_cb, self.srt_cb,
                        ],
                    ),
                    self._build_output_dir_row(),
                    ft.Row(controls=[self.diarize_cb, self.fast_cb], wrap=True),
                ],
            ),
        )

    def _build_output_dir_row(self) -> ft.Row:
        p = self.palette
        self.out_label = ft.Text(
            "Automatic: next to input files",
            color=p["text_dim"], size=12, expand=True, no_wrap=True,
        )
        return ft.Row(
            spacing=8,
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self.out_label,
                ft.OutlinedButton(
                    "Choose",
                    icon=ft.Icons.FOLDER_OUTLINED,
                    on_click=self._pick_output_folder,
                    style=ft.ButtonStyle(color=p["text_secondary"]),
                ),
            ],
        )

    def _build_action_card(self) -> ft.Container:
        p = self.palette

        self.progress_bar = ft.ProgressBar(
            value=0, color=p["accent_a"], bgcolor=p["surface_alt"],
            height=6, border_radius=3,
        )

        self.start_btn = ft.ElevatedButton(
            content=ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
                controls=[
                    ft.Icon(ft.Icons.PLAY_ARROW_ROUNDED, color=ft.Colors.WHITE, size=20),
                    ft.Text("Start Transcription",
                            color=ft.Colors.WHITE,
                            weight=ft.FontWeight.W_600,
                            font_family=self.bold_font),
                ],
            ),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=ft.Padding.symmetric(horizontal=24, vertical=18),
                shadow_color=p["accent_a"] + "60",
                elevation=8,
            ),
            on_click=self._start,
            disabled=True,
        )

        # Apply gradient via a wrapping container hack - ElevatedButton's bg
        # is a solid color in Flet, so we use a Container gradient behind a
        # transparent button for the visual.
        gradient_btn = ft.Container(
            border_radius=12,
            gradient=ft.LinearGradient(
                colors=[p["accent_a"], p["accent_b"]],
                begin=ft.Alignment.CENTER_LEFT,
                end=ft.Alignment.CENTER_RIGHT,
            ),
            shadow=ft.BoxShadow(blur_radius=24, color=p["accent_a"] + "40",
                                offset=ft.Offset(0, 8)),
            content=self.start_btn,
            opacity=0.45,
        )
        self.start_button_wrap = gradient_btn

        self.cancel_btn = ft.OutlinedButton(
            "Cancel",
            icon=ft.Icons.STOP_ROUNDED,
            on_click=lambda _: self._cancel(),
            style=ft.ButtonStyle(
                color=p["text_secondary"],
                side=ft.BorderSide(1, p["border"]),
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=ft.Padding.symmetric(horizontal=20, vertical=16),
            ),
            disabled=True,
        )

        return ft.Container(
            padding=ft.Padding.all(20),
            border_radius=20,
            bgcolor=p["surface"],
            border=ft.Border.all(1, p["border"]),
            content=ft.Column(
                spacing=14,
                controls=[
                    self.progress_bar,
                    ft.Row(spacing=10, controls=[gradient_btn, self.cancel_btn]),
                ],
            ),
        )

    def _build_log_card(self) -> ft.Container:
        p = self.palette
        self.log_list = ft.Column(
            expand=True, spacing=2, scroll=ft.ScrollMode.AUTO,
        )
        return ft.Container(
            expand=True,
            padding=ft.Padding.all(16),
            border_radius=20,
            bgcolor=p["surface"],
            border=ft.Border.all(1, p["border"]),
            content=ft.Column(
                expand=True, spacing=8,
                controls=[
                    ft.Row(
                        controls=[
                            ft.Icon(ft.Icons.TERMINAL, size=14,
                                    color=p["text_secondary"]),
                            ft.Text("Log", size=12, weight=ft.FontWeight.W_600,
                                    color=p["text_secondary"]),
                        ],
                        spacing=6,
                    ),
                    ft.Container(
                        expand=True,
                        content=self.log_list,
                    ),
                ],
            ),
        )

    def _build_status_bar(self) -> ft.Container:
        p = self.palette
        cuda_state = "CUDA" if self.cuda_available else "CPU"
        self.status_label = ft.Text(
            f"Ready � {cuda_state}" if self.cuda_checked else "Checking transcription backend.",
            size=11, color=p["text_secondary"],
        )
        version_label = ft.Text(
            f"v{APP_VERSION}",
            size=11, color=p["text_dim"],
        )
        return ft.Container(
            padding=ft.Padding.symmetric(horizontal=4, vertical=4),
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[self.status_label, version_label],
            ),
        )

    def _build_loading_overlay(self) -> ft.Container:
        p = self.palette
        self.loading_text = ft.Text(
            "Loading model.", size=14, color=p["text_primary"],
        )
        return ft.Container(
            visible=False,
            expand=True,
            bgcolor="#0A0E2799",
            alignment=ft.Alignment.CENTER,
            content=ft.Column(
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
                controls=[
                    ft.ProgressRing(color=p["accent_a"], width=48, height=48),
                    self.loading_text,
                ],
            ),
        )

    # ----- file management -----
    async def _pick_files(self, _):
        files = await self.file_picker.pick_files(
            allow_multiple=True,
            file_type=ft.FilePickerFileType.CUSTOM,
            allowed_extensions=[ext.lstrip(".") for ext in SUPPORTED_EXTS],
            dialog_title="Select audio/video files",
        )
        self._add_paths([f.path for f in (files or []) if f.path])

    async def _pick_input_folder(self, _):
        path = await self.file_picker.get_directory_path(
            dialog_title="Select folder with audio/video"
        )
        if path:
            self._add_paths([path])

    async def _pick_output_folder(self, _):
        path = await self.file_picker.get_directory_path(
            dialog_title="Select output folder"
        )
        if path:
            self._set_output_dir(path)

    def _add_paths(self, paths: list[str]):
        added = 0
        for p in paths:
            if os.path.isdir(p):
                # walk folder
                for root, _, files in os.walk(p):
                    for f in files:
                        fp = os.path.join(root, f)
                        if os.path.splitext(fp)[1].lower() in SUPPORTED_EXTS:
                            if self._add_file(fp):
                                added += 1
            elif os.path.isfile(p):
                if self._add_file(p):
                    added += 1
        if added:
            self._log("info", f"Added {added} file(s).")
            self._update_start_enabled()

    def _add_file(self, path: str) -> bool:
        # Skip dupes
        if any(f["path"] == path for f in self.files):
            return False
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        entry = {
            "path": path, "name": os.path.basename(path),
            "size": size, "status": "Queued", "pct": 0.0,
        }
        self.files.append(entry)
        self._render_file_row(entry)
        return True

    def _clear_files(self, _=None):
        self.files.clear()
        self.file_rows.clear()
        self.file_list_col.controls.clear()
        self.file_list_col.update()
        self._update_start_enabled()
        self._log("info", "Cleared file list.")

    def _set_output_dir(self, path: str):
        self.output_dir = path
        self.out_label.value = path
        self.out_label.color = self.palette["text_primary"]
        self.out_label.update()
        self._update_start_enabled()
        self._log("info", f"Output folder: {path}")

    def _update_start_enabled(self):
        self.empty_state.visible = not self.files
        self.file_stack.update()
        enabled = bool(self.files) and not self.running and self.backend_ready and not self.import_error
        self.start_btn.disabled = not enabled
        self.start_button_wrap.opacity = 1.0 if enabled else 0.45
        self.start_button_wrap.update()
        self.out_label.value = self.output_dir or (str(self._default_output_dir()) if self.files else "Automatic: next to input files")
        self.out_label.update()

    def _default_output_dir(self):
        return Path(self.files[0]["path"]).resolve().parent / "Transcripts"

    def _render_file_row(self, entry: dict):
        p = self.palette
        row = self._build_file_row(entry)
        self.file_rows[entry["path"]] = row
        self.file_list_col.controls.append(row)
        self.file_list_col.update()

    def _build_file_row(self, entry: dict) -> ft.Container:
        p = self.palette
        name_text = ft.Text(
            entry["name"], size=13, weight=ft.FontWeight.W_500,
            color=p["text_primary"], no_wrap=True, expand=True,
        )
        size_text = ft.Text(
            fmt_size(entry["size"]), size=11, color=p["text_secondary"],
        )
        status_pill = ft.Container(
            padding=ft.Padding.symmetric(horizontal=10, vertical=4),
            border_radius=10,
            bgcolor=p["surface_alt"],
            content=ft.Text(
                entry["status"], size=11, color=p["text_secondary"],
            ),
        )
        progress = ft.ProgressBar(
            value=0, color=p["accent_a"], bgcolor=p["surface_alt"],
            height=3, border_radius=2, expand=True,
        )

        # Store refs for live updates
        entry["_status_pill"] = status_pill
        entry["_progress"] = progress
        entry["_name"] = name_text

        def remove_self(_):
            self.files = [f for f in self.files if f["path"] != entry["path"]]
            self.file_rows.pop(entry["path"], None)
            if entry["_container_ref"]:
                self.file_list_col.controls.remove(entry["_container_ref"])
                self.file_list_col.update()
            self._update_start_enabled()

        entry["_container_ref"] = None  # set below
        remove_btn = ft.IconButton(
            icon=ft.Icons.CLOSE, icon_size=16,
            icon_color=p["text_dim"], tooltip="Remove",
            on_click=remove_self,
        )

        container = ft.Container(
            padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            border_radius=12,
            bgcolor=p["surface_alt"],
            animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            content=ft.Column(
                spacing=6,
                controls=[
                    ft.Row(
                        controls=[name_text, size_text, status_pill, remove_btn],
                        spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    progress,
                ],
            ),
        )
        entry["_container_ref"] = container
        return container

    # ----- transcription control -----
    def _start(self, _=None):
        if not self.files or self.running or self.import_error or not self.backend_ready:
            return
        try:
            output = Path(self.output_dir) if self.output_dir else self._default_output_dir()
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._show_error_dialog("Cannot create output folder", str(exc))
            return
        try:
            self._launch_batch(str(output))
        except Exception as exc:
            self.running = False
            self._hide_loading()
            self._update_start_enabled()
            self.cancel_btn.disabled = True
            self.cancel_btn.update()
            self._show_error_dialog("Cannot start transcription", str(exc))

    def _launch_batch(self, output):
        self.running = True
        self.batch_failed = False
        self._update_start_enabled()
        self.cancel_btn.disabled = False
        self.cancel_btn.update()
        self.progress_bar.value = 0
        self.progress_bar.update()
        self._log("info", "Starting transcription.")
        self._show_loading(f"Loading model '{self.model_dd.value}'.")

        formats: list[str] = []
        if self.txt_cb.value: formats.append("txt")
        if self.srt_cb.value: formats.append("srt")
        if not formats: formats = ["txt"]

        lang_name = self.lang_dd.value
        lang_code = None
        for label, code in LANGUAGES:
            if label == lang_name:
                lang_code = code
                break

        self.transcriber = BatchTranscriber(
            model_size=self.model_dd.value,
            device=self.device_dd.value,
            compute_type="float16" if self.device_dd.value == "cuda" else "int8",
            language=lang_code,
            formats=tuple(formats),
            queue=self.queue,
            diarize=bool(self.diarize_cb.value),
            fast=bool(self.fast_cb.value),
        )
        # Switch UI to per-file progress
        for entry in self.files:
            entry["status"] = "Queued"
            entry["pct"] = 0.0
            self._refresh_file_entry(entry)

        self.worker = threading.Thread(
            target=self._run_batch,
            args=([f["path"] for f in self.files], output),
            daemon=True,
        )
        self.worker.start()

    def _run_batch(self, paths: list[str], out_dir: str):
        try:
            self.transcriber.transcribe_batch(paths, out_dir)
        except Exception as e:
            self.queue.put({"kind": "error", "msg": f"Worker crashed: {e}"})
        finally:
            self.queue.put({"kind": "_done_internal", "msg": ""})

    def _cancel(self):
        if self.transcriber:
            self.transcriber.cancel()
            self._log("warn", "Cancellation requested.")

    # ----- queue polling -----
    def _start_poll(self):
        async def poll():
            while True:
                try:
                    msg = self.queue.get_nowait()
                    self._handle_message(msg)
                except queue.Empty:
                    await asyncio.sleep(0.12)
                except Exception as exc:
                    self._on_page_error(exc)
                    await asyncio.sleep(0.12)
        self.page.run_task(poll)

    def _handle_message(self, msg: dict):
        kind = msg.get("kind")
        text = msg.get("msg", "")
        if kind in ("log", "error", "_done_internal"):
            if kind == "error":
                self.batch_failed = True
                self._log("error", text)
                if self.loading_overlay.visible:
                    self._hide_loading()
                self._show_error_dialog("Transcription error", text)
            elif kind == "_done_internal":
                # bookkeeping only
                self._on_worker_finished()
            else:
                # Detect model-load message
                if text.startswith("Loading model"):
                    self._show_loading(text)
                elif text.startswith("Model loaded") or text == "Model loaded on CPU.":
                    self._hide_loading()
                    if text == "Model loaded on CPU.":
                        self.device_dd.value = "cpu"
                        self.device_dd.update()
                self._log("info", text)
        elif kind == "progress":
            # msg has: file_index, total_files, pct
            idx = msg.get("file_index", 0)
            total = msg.get("total_files", 0)
            pct = msg.get("pct", 0.0)
            if 0 < idx <= len(self.files):
                entry = self.files[idx - 1]
                entry["status"] = "Transcribing" if pct < 100 else "Done"
                entry["pct"] = pct
                self._refresh_file_entry(entry)
            if total > 0:
                self.progress_bar.value = (idx / total)
                self.progress_bar.update()
            self.status_label.value = text
            self.status_label.update()
            self._log("progress", text)
        elif kind == "file_error":
            self.batch_failed = True
            idx = msg.get("file_index", 0)
            if 0 < idx <= len(self.files):
                entry = self.files[idx - 1]
                entry["status"] = "Error"
                self._refresh_file_entry(entry)
            self._hide_loading()
            self._log("error", text)
        elif kind == "done":
            self._log("warn" if self.batch_failed else "success", text)

    def _on_worker_finished(self):
        self.running = False
        self._hide_loading()
        # Reset progress to full
        cancelled = bool(self.transcriber and self.transcriber._cancel.is_set())
        if not self.batch_failed and not cancelled:
            self.progress_bar.value = 1.0
        self.progress_bar.update()
        # Mark any still-Queued as Done (defensive)
        for entry in self.files:
            if entry["status"] in ("Queued", "Transcribing"):
                entry["status"] = "Cancelled" if cancelled else "Error"
                self._refresh_file_entry(entry)
        self._update_start_enabled()
        self.cancel_btn.disabled = True
        self.cancel_btn.update()
        self.status_label.value = "Cancelled" if cancelled else ("Finished with errors" if self.batch_failed else "Done")
        self.status_label.update()
        self._log("warn" if self.batch_failed or cancelled else "success", "--- Finished ---")

    def _refresh_file_entry(self, entry: dict):
        pill = entry.get("_status_pill")
        progress = entry.get("_progress")
        if pill:
            p = self.palette
            color_map = {
                "Queued": (p["surface_alt"], p["text_secondary"]),
                "Transcribing": (p["accent_a"] + "30", p["accent_a"]),
                "Done": (p["success"] + "30", p["success"]),
                "Error": (p["error"] + "30", p["error"]),
            }
            bg, fg = color_map.get(entry["status"], (p["surface_alt"], p["text_secondary"]))
            pill.bgcolor = bg
            pill.content.value = entry["status"]
            pill.content.color = fg
            pill.update()
        if progress:
            progress.value = entry.get("pct", 0) / 100
            progress.update()

    # ----- log -----
    def _log(self, level: str, text: str):
        if not self.log_list:
            return
        p = self.palette
        ts = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "info": p["text_secondary"],
            "progress": p["accent_a"],
            "success": p["success"],
            "error": p["error"],
            "warn": p["warning"],
        }
        bullet = "?" if level in ("error", "success", "warn") else ">"
        line = ft.Row(
            spacing=6,
            controls=[
                ft.Text(ts, size=11, color=p["text_dim"],
                        font_family=self.bold_font),
                ft.Text(bullet, size=11, color=color_map.get(level, p["text_secondary"])),
                ft.Text(text, size=12, color=color_map.get(level, p["text_primary"]),
                        selectable=True, expand=True, no_wrap=False),
            ],
        )
        self.log_list.controls.append(line)
        # Keep the log bounded - drop oldest lines past 500
        if len(self.log_list.controls) > 500:
            self.log_list.controls = self.log_list.controls[-500:]
        self.log_list.update()

    # ----- loading overlay -----
    def _show_loading(self, text: str):
        if not self.loading_overlay:
            return
        self.loading_text.value = text
        self.loading_overlay.visible = True
        self.loading_overlay.update()
        # Animate the trailing dots
        if not getattr(self, "_loading_animating", False):
            self._loading_animating = True
            self._loading_text_base = text
            self._animate_loading_dots()

    def _animate_loading_dots(self):
        if not self.loading_overlay or not self.loading_overlay.visible:
            self._loading_animating = False
            return
        base = getattr(self, "_loading_text_base", self.loading_text.value or "Loading")
        n = (int(time.time() * 2) % 4)
        self.loading_text.value = base + ("." * n)
        try:
            self.loading_text.update()
        except Exception:
            pass
        # Schedule next frame (~500ms cadence)
        try:
            self.page.after(500, self._animate_loading_dots)
        except Exception:
            self._loading_animating = False

    def _hide_loading(self):
        if not self.loading_overlay:
            return
        self.loading_overlay.visible = False
        self._loading_animating = False
        self.loading_overlay.update()

    # ----- errors -----
    def _on_page_error(self, e):
        self._log("error", f"Page error: {e}")

    def _show_error_dialog(self, title: str, body: str):
        dlg = ft.AlertDialog(
            title=ft.Text(title, color=self.palette["text_primary"]),
            content=ft.Text(body, color=self.palette["text_secondary"],
                            selectable=True),
            actions=[
                ft.TextButton("OK", on_click=lambda _: self.page.pop_dialog()),
            ],
        )
        self.page.show_dialog(dlg)

    # ----- theme rebuild -----
    def _rebuild_theme(self):
        # The simplest reliable path: re-create the controls with the new palette.
        # Easiest: clear the page and rebuild. (Cheap because the app state is
        # already in self.* and the Flet tree is small.)
        for child in list(self.page.controls):
            self.page.remove(child)
        self.palette = PALETTES[self.theme_name]
        self.page.bgcolor = self.palette["bg_a"]
        self.page.theme_mode = (
            ft.ThemeMode.DARK if self.theme_name == "dark" else ft.ThemeMode.LIGHT
        )
        self._build_ui()
        self.file_rows.clear()
        for entry in self.files:
            self._render_file_row(entry)
        if self.output_dir:
            self.out_label.value = self.output_dir
            self.out_label.color = self.palette["text_primary"]
        self._update_start_enabled()
        self.page.update()


def main(page: ft.Page):
    try:
        App(page)
    except Exception as e:
        # Last-ditch error display
        page.add(
            ft.Container(
                padding=40, expand=True,
                content=ft.Column(
                    controls=[
                        ft.Icon(ft.Icons.ERROR_OUTLINE, size=64,
                                color=ft.Colors.RED),
                        ft.Text("Failed to start BatchTranscribe",
                                size=20, weight=ft.FontWeight.BOLD),
                        ft.Text(str(e), selectable=True, size=12),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
            )
        )


if __name__ == "__main__":
    ft.run(main, assets_dir="assets")


