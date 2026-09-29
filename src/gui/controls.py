"""Control panel exposing every mandatory specification parameter.

Each control edits the JSON configuration and nothing else -- the panel builds an override
mapping and hands it back, so the run it starts is exactly the run the CLI would perform with the
same scenario file. No widget computes anything about tracking.

Specification parameter coverage
--------------------------------
Walked against the 25-row table in ``docs/PROBLEM_STATEMENT.md``:

============  =========================  =================================================
Parameter     Control                    Note
============  =========================  =================================================
1  screen     Scene width/height         2000x2000 minimum enforced by ``src.config``
2  camera     Monochrome (display only)  Colour display available in Mode B via colour_display
                                         toggle; the vision pipeline always receives grayscale
3  resolution Camera width/height
4  FOV        FOV horizontal/vertical
5  rate       Camera update rate         >= 30 Hz enforced by ``src.config``
6  initial    Camera initial position    centre / random
7  type       Beacon spot (fixed)        The specification fixes this; nothing to select
8  targets    Target count               1 mandatory; >1 now implemented (up to 4)
9  shape      Target shape               gaussian / square / circle
10 size       Target size                5-20 px enforced by ``src.config``
11 location   Target initial position    random / centre
12 motion     Motion type                all four mandatory plus three optional
13 pan speed  Max pan speed              5-10 deg/s enforced by ``src.config``
14 tilt speed Max tilt speed             5-10 deg/s enforced
15 control    Control update rate        >= 20 Hz enforced
16-20         Displayed, not controlled  Performance requirements, shown in the metrics panel
21 noise      Gaussian / Poisson / S&P   individually toggleable
22 noise sd   Gaussian sigma             <= 20 enforced
23 jitter     Camera jitter              <= 20 px/frame enforced
24 atmosphere Atmospheric preset         clear / haze / fog / rain / low light
25 platform   Platform motion type+rate  linear mandatory, four optional types
============  =========================  =================================================
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from src.config import AppConfig

__all__ = ["ControlPanel"]


class ControlPanel(QWidget):
    """Edits the run configuration. Produces override mappings; decides nothing else."""

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        """Build the panel from a configuration.

        Args:
            config: The configuration whose current values seed the controls.
            parent: Qt parent.
        """
        super().__init__(parent)
        self.config = config
        layout = QVBoxLayout(self)
        
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search settings... (Global Search)")
        self.search_bar.textChanged.connect(self._filter_settings)
        layout.addWidget(self.search_bar)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)

        self.mode_tab = self._build_mode_tab()
        self.camera_tab = self._build_camera_tab()
        self.target_tab = self._build_target_tab()
        self.noise_tab = self._build_noise_tab()
        
        tabs.addTab(self.mode_tab, "Mode Selection")
        tabs.addTab(self.camera_tab, "Camera")
        tabs.addTab(self.target_tab, "Target")
        tabs.addTab(self.noise_tab, "Noise")

        if hasattr(self, "mode"):
            self.mode.currentTextChanged.connect(self._update_mode_visibility)
            self._update_mode_visibility(self.mode.currentText())

    def _update_mode_visibility(self, mode_text: str) -> None:
        """Enable/disable controls based on the selected mode."""
        is_sim = (mode_text.lower() == "simulation")
        
        self.target_tab.setEnabled(is_sim)
        self.noise_tab.setEnabled(is_sim)
        
        # Camera options for simulation
        if hasattr(self, "pan_speed"):
            self.pan_speed.setEnabled(is_sim)
            self.tilt_speed.setEnabled(is_sim)
            self.camera_position.setEnabled(is_sim)
            self.scene_w.setEnabled(is_sim)
            self.scene_h.setEnabled(is_sim)
            
        # Video options for mode B
        if hasattr(self, "video_path"):
            self.video_path.setEnabled(not is_sim)
            self.truth_path.setEnabled(not is_sim)
            self.colour_display_check.setEnabled(not is_sim)

    def _set_layout_visible(self, layout, visible: bool) -> None:
        """Helper to hide/show all widgets in a nested layout."""
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().setVisible(visible)
            elif item.layout():
                self._set_layout_visible(item.layout(), visible)

    def _filter_settings(self, text: str) -> None:
        """Filter settings rows across all tabs based on the search text."""
        query = text.lower()
        first_match_idx = -1
        tabs_widget = self.findChild(QTabWidget)
        
        tab_list = [self.mode_tab, self.camera_tab, self.target_tab, self.noise_tab]
        for tab_idx, tab in enumerate(tab_list):
            has_match_in_tab = False
            layouts_to_check = []
            
            if tab.layout():
                layouts_to_check.append(tab.layout())
                for i in range(tab.layout().count()):
                    item = tab.layout().itemAt(i)
                    if item and item.layout():
                        layouts_to_check.append(item.layout())
                        
            for form in layouts_to_check:
                if isinstance(form, QFormLayout):
                    for i in range(form.rowCount()):
                        label_item = form.itemAt(i, QFormLayout.ItemRole.LabelRole)
                        field_item = form.itemAt(i, QFormLayout.ItemRole.FieldRole)
                        
                        has_match = False
                        if label_item and label_item.widget():
                            has_match = query in label_item.widget().text().lower()
                        if field_item and field_item.widget() and hasattr(field_item.widget(), "text"):
                            has_match = has_match or (query in field_item.widget().text().lower())
                        
                        visible = has_match or not query
                        if has_match and query:
                            has_match_in_tab = True
                        
                        if label_item and label_item.widget():
                            label_item.widget().setVisible(visible)
                        if field_item:
                            if field_item.widget():
                                field_item.widget().setVisible(visible)
                            elif field_item.layout():
                                self._set_layout_visible(field_item.layout(), visible)
                                
            if has_match_in_tab and first_match_idx == -1:
                first_match_idx = tab_idx
                
        if query and first_match_idx != -1 and tabs_widget:
            tabs_widget.setCurrentIndex(first_match_idx)

    # -- tab builders -------------------------------------------------------------------------

    def _spin(self, value: float, low: float, high: float, step: float = 1.0,
              decimals: int = 0) -> QDoubleSpinBox:
        """Return a configured spin box."""
        box = QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        box.setValue(value)
        return box

    def _build_target_tab(self) -> QWidget:
        """Target count, shape, size, initial location and motion."""
        widget = QWidget()
        form = QFormLayout(widget)
        target = self.config.target

        self.target_count = QSpinBox()
        self.target_count.setRange(1, 4)
        self.target_count.setValue(target.count)
        self.target_count.setToolTip(
            "One target is mandatory; up to 4 simultaneous targets are supported "
            "as an optional bonus feature.")
        form.addRow("Targets  [1–4]", self.target_count)

        self.target_shape = QComboBox()
        self.target_shape.addItems(["Gaussian", "Square", "Circle"])
        self.target_shape.setCurrentText(target.shape.capitalize())
        form.addRow("Shape", self.target_shape)

        self.target_size = self._spin(float(target.size_px), 5, 20)
        form.addRow("Size  [5–20 px]", self.target_size)

        self.target_position = QComboBox()
        self.target_position.addItems(["Random", "Center"])
        pos = target.initial_position if target.initial_position in ("random", "center") else "random"
        self.target_position.setCurrentText(pos.capitalize())
        self.target_position.setToolTip(
            "Random: beacon spawns at a random canvas position.\n"
            "Center: beacon spawns at the centre of the canvas.\n"
            "For a fixed custom position use a scenario JSON override.")
        form.addRow("Initial location", self.target_position)

        self.motion = QComboBox()
        self.motion.addItems(["Linear", "Circular", "Figure8", "Random",
                              "Spiral", "Sinusoidal", "Ornstein_uhlenbeck", "Custom"])
        self.motion.setCurrentText(target.motion_type.capitalize())
        self.motion.setToolTip(
            "Mandatory: linear, circular, figure8, random.\n"
            "Optional: spiral, sinusoidal, ornstein_uhlenbeck, custom.")
        form.addRow("Motion type", self.motion)

        self.boundary = QComboBox()
        self.boundary.addItems(["Bounce", "Wrap", "Clamp"])
        self.boundary.setCurrentText(target.boundary_behaviour.capitalize())
        form.addRow("Edge behaviour", self.boundary)
        return widget

    def _build_camera_tab(self) -> QWidget:
        """Scene size, camera resolution, FOV, rates and slew limits."""
        widget = QWidget()
        form = QFormLayout(widget)
        camera = self.config.camera
        scene = self.config.scene

        self.scene_w = self._spin(scene.width, 2000, 8000, 100)
        self.scene_h = self._spin(scene.height, 2000, 8000, 100)
        form.addRow("Scene width  [min 2000]", self.scene_w)
        form.addRow("Scene height  [min 2000]", self.scene_h)

        form.addRow("Camera type", QLabel("Monochrome FPA  (colour display available in Mode B)"))

        self.res_w = self._spin(camera.resolution_width, 160, 2000, 10)
        self.res_h = self._spin(camera.resolution_height, 120, 2000, 10)
        form.addRow("Resolution width", self.res_w)
        form.addRow("Resolution height", self.res_h)

        self.fov_h = self._spin(camera.fov_horizontal_deg, 0.5, 30.0, 0.5, 2)
        self.fov_v = self._spin(camera.fov_vertical_deg, 0.5, 30.0, 0.5, 2)
        form.addRow("FOV horizontal (deg)", self.fov_h)
        form.addRow("FOV vertical (deg)", self.fov_v)

        self.camera_rate = self._spin(camera.update_rate_hz, 30, 120, 5)
        form.addRow("Camera rate  [min 30 Hz]", self.camera_rate)

        self.camera_position = QComboBox()
        self.camera_position.addItems(["Center", "Random"])
        cam_pos = camera.initial_position if camera.initial_position in ("center", "random") else "center"
        self.camera_position.setCurrentText(cam_pos.capitalize())
        self.camera_position.setToolTip(
            "Use Arrow Keys to pan the camera manually, or drag the camera in the viewport.")
        form.addRow("Initial position", self.camera_position)

        self.pan_speed = self._spin(camera.max_pan_speed_deg_s, 5.0, 10.0, 0.5, 1)
        self.tilt_speed = self._spin(camera.max_tilt_speed_deg_s, 5.0, 10.0, 0.5, 1)
        form.addRow("Max pan speed  [5–10 °/s]", self.pan_speed)
        form.addRow("Max tilt speed  [5–10 °/s]", self.tilt_speed)

        self.control_rate = self._spin(self.config.control.update_rate_hz, 20, 120, 5)
        form.addRow("Control rate  [min 20 Hz]", self.control_rate)
        return widget

    def _build_noise_tab(self) -> QWidget:
        """Sensor noise, atmosphere, jitter and platform motion."""
        widget = QWidget()
        form = QFormLayout(widget)
        noise = self.config.noise

        self.noise_enabled = QCheckBox("Noise pipeline enabled")
        self.noise_enabled.setChecked(noise.enabled)
        form.addRow(self.noise_enabled)

        self.gaussian_on = QCheckBox("Gaussian noise")
        self.gaussian_on.setChecked(bool(noise.gaussian.get("enabled", True)))
        self.gaussian_sigma = self._spin(float(noise.gaussian.get("sigma", 10.0)), 0, 20, 1, 1)
        form.addRow(self.gaussian_on, self.gaussian_sigma)
        form.addRow(QLabel("  Sigma  [max 20]"))

        self.poisson_on = QCheckBox("Poisson noise")
        self.poisson_on.setChecked(bool(noise.poisson.get("enabled", False)))
        form.addRow(self.poisson_on)

        self.sp_on = QCheckBox("Salt & pepper")
        self.sp_on.setChecked(bool(noise.salt_pepper.get("enabled", False)))
        self.sp_density = self._spin(float(noise.salt_pepper.get("density", 0.05)), 0, 0.10,
                                     0.01, 3)
        form.addRow(self.sp_on, self.sp_density)

        self.atmosphere = QComboBox()
        self.atmosphere.addItems(["Clear", "Haze", "Fog", "Rain", "Low_light"])
        self.atmosphere.setCurrentText(str(noise.atmospheric.get("preset", "clear")).capitalize())
        form.addRow("Atmosphere preset", self.atmosphere)

        self.jitter_on = QCheckBox("Camera jitter")
        self.jitter_on.setChecked(bool(noise.camera_jitter.get("enabled", True)))
        self.jitter_px = self._spin(float(noise.camera_jitter.get("max_px_per_frame", 5.0)),
                                    0, 20, 1, 1)
        form.addRow(self.jitter_on, self.jitter_px)
        form.addRow(QLabel("  Max px/frame  [max 20]"))

        self.platform_on = QCheckBox("Platform motion")
        self.platform_on.setChecked(bool(noise.platform_motion.get("enabled", False)))
        self.platform_type = QComboBox()
        self.platform_type.addItems(["Linear", "Circular", "Random", "Spiral", "Figure8"])
        self.platform_type.setCurrentText(str(noise.platform_motion.get("type", "linear")).capitalize())
        self.platform_px = self._spin(float(noise.platform_motion.get("max_px_per_frame", 5.0)),
                                      0, 20, 1, 1)
        form.addRow(self.platform_on, self.platform_type)
        form.addRow("  Max px/frame", self.platform_px)

        self.turbulence_on = QCheckBox("Turbulence / scintillation")
        self.turbulence_on.setChecked(bool(noise.turbulence.get("enabled", False)))
        form.addRow(self.turbulence_on)
        return widget

    def _build_mode_tab(self) -> QWidget:
        """Mode A / Mode B selection with a file picker for evaluator video."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        form = QFormLayout()

        self.mode = QComboBox()
        self.mode.addItems(["Simulation", "Video"])
        self.mode.setCurrentText(self.config.run.mode.capitalize())
        form.addRow("Mode", self.mode)

        self.duration = self._spin(self.config.run.duration_seconds, 1, 600, 1)
        form.addRow("Duration (seconds)", self.duration)

        self.seed = QSpinBox()
        self.seed.setRange(0, 10**6)
        self.seed.setValue(self.config.run.random_seed)
        form.addRow("Random seed", self.seed)

        picker = QHBoxLayout()
        self.video_path = QLineEdit(self.config.video_input.path or "")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._pick_video)
        picker.addWidget(self.video_path)
        picker.addWidget(browse)
        form.addRow("Video file (Mode B)", picker)

        truth_picker = QHBoxLayout()
        self.truth_path = QLineEdit(self.config.video_input.ground_truth_path or "")
        browse_truth = QPushButton("Browse...")
        browse_truth.clicked.connect(self._pick_truth)
        truth_picker.addWidget(self.truth_path)
        truth_picker.addWidget(browse_truth)
        form.addRow("Ground truth CSV", truth_picker)

        self.colour_display_check = QCheckBox("Colour display (Mode B)")
        self.colour_display_check.setChecked(self.config.video_input.colour_display)
        self.colour_display_check.setToolTip(
            "When enabled, the GUI displays colour video frames in Mode B. "
            "The vision pipeline always receives grayscale; this only affects what you see.")
        form.addRow(self.colour_display_check)

        layout.addLayout(form)

        # Fixed: was using \\n (literal backslash-n) instead of \n; also added word wrap.
        _mode_note = QLabel(
            "Mode B bypasses the virtual PTZ camera: the video is the scene, "
            "so pan/tilt is disabled and angular metrics are omitted rather than fabricated.")
        _mode_note.setWordWrap(True)
        _mode_note.setStyleSheet("color: #8b949e; font-size: 9pt; padding: 4px;")
        layout.addWidget(_mode_note)
        return widget

    def _pick_video(self) -> None:
        """Open a file dialog for the Mode B input video."""
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "",
                                              "Video (*.mp4 *.avi *.mov);;All files (*)")
        if path:
            self.video_path.setText(path)
            self.mode.setCurrentText("video")

    def _pick_truth(self) -> None:
        """Open a file dialog for an optional ground-truth sidecar."""
        path, _ = QFileDialog.getOpenFileName(self, "Select ground truth", "",
                                              "CSV (*.csv);;All files (*)")
        if path:
            self.truth_path.setText(path)

    # -- output -------------------------------------------------------------------------------

    def overrides(self) -> Dict[str, Any]:
        """Return the control values as a configuration override mapping.

        The same shape as a scenario file, so a GUI run and a CLI run with that file are the
        same run.

        Returns:
            A partial configuration document.
        """
        return {
            "run": {
                "mode": self.mode.currentText().lower(),
                "duration_seconds": float(self.duration.value()),
                "random_seed": int(self.seed.value()),
                "headless": False,
            },
            "scene": {"width": int(self.scene_w.value()), "height": int(self.scene_h.value())},
            "camera": {
                "resolution_width": int(self.res_w.value()),
                "resolution_height": int(self.res_h.value()),
                "fov_horizontal_deg": float(self.fov_h.value()),
                "fov_vertical_deg": float(self.fov_v.value()),
                "update_rate_hz": float(self.camera_rate.value()),
                "initial_position": self.camera_position.currentText().lower(),
                "max_pan_speed_deg_s": float(self.pan_speed.value()),
                "max_tilt_speed_deg_s": float(self.tilt_speed.value()),
            },
            "target": {
                "count": int(self.target_count.value()),
                "shape": self.target_shape.currentText().lower(),
                "size_px": int(self.target_size.value()),
                "initial_position": self.target_position.currentText().lower(),
                "boundary_behaviour": self.boundary.currentText().lower(),
                "motion": {"type": self.motion.currentText().lower()},
            },
            "noise": {
                "enabled": self.noise_enabled.isChecked(),
                "gaussian": {"enabled": self.gaussian_on.isChecked(),
                             "sigma": float(self.gaussian_sigma.value())},
                "poisson": {"enabled": self.poisson_on.isChecked()},
                "salt_pepper": {"enabled": self.sp_on.isChecked(),
                                "density": float(self.sp_density.value())},
                "atmospheric": {"preset": self.atmosphere.currentText().lower()},
                "camera_jitter": {"enabled": self.jitter_on.isChecked(),
                                  "max_px_per_frame": float(self.jitter_px.value())},
                "platform_motion": {"enabled": self.platform_on.isChecked(),
                                    "type": self.platform_type.currentText().lower(),
                                    "max_px_per_frame": float(self.platform_px.value())},
                "turbulence": {"enabled": self.turbulence_on.isChecked()},
            },
            "control": {"update_rate_hz": float(self.control_rate.value())},
            "video_input": {
                "path": self.video_path.text().strip() or None,
                "ground_truth_path": self.truth_path.text().strip() or None,
                "colour_display": self.colour_display_check.isChecked(),
            },
        }
