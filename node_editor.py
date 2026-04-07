import sys, uuid, copy
from PyQt6.QtWidgets import (
    QApplication, QGraphicsScene, QGraphicsView, QGraphicsItem, QGraphicsPathItem,
    QWidget, QVBoxLayout, QHBoxLayout, QMenu, QFileDialog, QPushButton,
    QLabel, QTextEdit, QGraphicsProxyWidget, QSpinBox, QLineEdit,
    QComboBox, QSlider, QCheckBox, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import (QColor, QPen, QBrush, QPainterPath, QFont, QPainter,
                         QKeySequence, QUndoStack, QUndoCommand)

# ─── Theme ───────────────────────────────────────────────────────────
class Theme:
    current = "dark"
    colors = {
        "dark": {
            "bg": "#1E1E2E", "grid": "#313244", "node_bg": "#181825",
            "node_border": "#45475A", "node_sel": "#89B4FA", "text": "#CDD6F4",
            "edge": "#A6ADC8", "edge_glow": QColor(137, 180, 250, 40),
            "shadow": QColor(0, 0, 0, 80), "widget_bg": "#313244",
            "btn_bg": "#45475A", "btn_hover": "#585B70", "btn_text": "#CDD6F4",
            "boundary": QColor(255, 80, 80, 40),
        }
    }
    @classmethod
    def get(cls, key):
        c = cls.colors[cls.current][key]
        return QColor(c) if isinstance(c, str) else c
    @classmethod
    def get_str(cls, key):
        return cls.colors[cls.current][key]

NODE_COLORS = {
    "Source": QColor("#A6E3A1"), "Vector DB": QColor("#89B4FA"),
    "LLM": QColor("#CBA6F7"), "Input": QColor("#FAB387"),
    "Output": QColor("#F38BA8"), "Memory": QColor("#F9E2AF"),
    "Processor": QColor("#94E2D5"), "Logic": QColor("#F5C2E7"),
    "Utility": QColor("#CDD6F4"), "Prompt": QColor("#B4BEFE"),
    "Retrieval": QColor("#74C7EC"), "Debug": QColor("#A6ADC8"),
}

# ─── Full Node Specifications ────────────────────────────────────────
NODE_SPEC = {
    "PDF Loader": {
        "cat": "Source", "in": 0, "out": 1, "priority": 1,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Load PDF and output document chunks",
        "in_labels": [], "out_labels": ["docs"],
        "config_options": {},
    },
    "FAISS DB": {
        "cat": "Vector DB", "in": 1, "out": 1, "priority": 4,
        "stateful": True, "shared_state_allowed": True,
        "supports_branching": False, "supports_loop": False,
        "description": "Retrieve similar chunks based on query",
        "in_labels": ["query"], "out_labels": ["chunks"],
        "config_options": {"k": 4},
    },
    "Ollama LLM": {
        "cat": "LLM", "in": 1, "out": 1, "priority": 10,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Generate answer from prompt",
        "in_labels": ["prompt"], "out_labels": ["answer"],
        "config_options": {},
    },
    "Query Input": {
        "cat": "Input", "in": 0, "out": 1, "priority": 0,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Entry point for user query",
        "in_labels": [], "out_labels": ["query"],
        "config_options": {},
    },
    "Response Output": {
        "cat": "Output", "in": 1, "out": 0, "priority": 100,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Display final response",
        "in_labels": ["result"], "out_labels": [],
        "config_options": {},
    },
    "Buffer": {
        "cat": "Memory", "in": 1, "out": 1, "priority": 3,
        "stateful": True, "shared_state_allowed": True,
        "supports_branching": False, "supports_loop": True,
        "description": "Store conversation history",
        "in_labels": ["data"], "out_labels": ["memory"],
        "config_options": {"mode": "shared", "window_size": 5},
    },
    "Cache": {
        "cat": "Memory", "in": 1, "out": 2, "priority": 2,
        "stateful": True, "shared_state_allowed": True,
        "supports_branching": True, "supports_loop": False,
        "description": "Fast query-answer reuse. Short-circuits on hit.",
        "in_labels": ["query"], "out_labels": ["hit", "miss"],
        "config_options": {"ttl_seconds": 3600, "max_entries": 100},
    },
    "Merge": {
        "cat": "Processor", "in": 2, "out": 1, "priority": 6,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Combine query, memory, and retrieved context",
        "in_labels": ["a", "b"], "out_labels": ["merged"],
        "config_options": {},
    },
    "Copy": {
        "cat": "Utility", "in": 1, "out": 2, "priority": 8,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": True, "supports_loop": False,
        "description": "Duplicate data for branching",
        "in_labels": ["data"], "out_labels": ["out_a", "out_b"],
        "config_options": {},
    },
    "Router": {
        "cat": "Logic", "in": 1, "out": 2, "priority": 5,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": True, "supports_loop": False,
        "description": "Conditional routing",
        "in_labels": ["data"], "out_labels": ["route_a", "route_b"],
        "config_options": {"routing_type": "keyword"},
    },
    # ── NEW NODES ──────────────────────────────────────────────────────
    "Prompt Template": {
        "cat": "Prompt", "in": 3, "out": 1, "priority": 9,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Format query+context+memory into a prompt string",
        "in_labels": ["query", "context", "memory"],
        "out_labels": ["prompt"],
        "config_options": {"template": ""},
    },
    "System Message": {
        "cat": "Prompt", "in": 0, "out": 1, "priority": 8,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Constant system instruction emitted at graph start",
        "in_labels": [], "out_labels": ["system"],
        "config_options": {"message": ""},
    },
    "Reranker": {
        "cat": "Retrieval", "in": 1, "out": 1, "priority": 6,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Re-rank retrieved chunks by relevance score, keep top_k",
        "in_labels": ["chunks"], "out_labels": ["chunks"],
        "config_options": {"top_k": 3},
    },
    "Memory Formatter": {
        "cat": "Memory", "in": 1, "out": 1, "priority": 5,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Condense/truncate conversation history before feeding LLM",
        "in_labels": ["memory"], "out_labels": ["memory"],
        "config_options": {"max_tokens": 500, "summarize": False},
    },
    "Conversation Starter": {
        "cat": "Memory", "in": 0, "out": 1, "priority": 1,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Provides initial system context before the first query",
        "in_labels": [], "out_labels": ["memory"],
        "config_options": {"text": ""},
    },
    "Score Filter": {
        "cat": "Retrieval", "in": 1, "out": 1, "priority": 6,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Drop retrieved chunks below a relevance threshold",
        "in_labels": ["chunks"], "out_labels": ["chunks"],
        "config_options": {"threshold": 0.5},
    },
    "Debug Inspector": {
        "cat": "Debug", "in": 1, "out": 1, "priority": 0,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Display data at this point in the graph; passes through unchanged",
        "in_labels": ["any"], "out_labels": ["pass-through"],
        "config_options": {},
    },
    "Multi Merge": {
        "cat": "Processor", "in": 4, "out": 1, "priority": 7,
        "stateful": False, "shared_state_allowed": False,
        "supports_branching": False, "supports_loop": False,
        "description": "Merge up to 4 inputs: system + memory + context + query",
        "in_labels": ["in_0", "in_1", "in_2", "in_3"],
        "out_labels": ["merged"],
        "config_options": {},
    },
    "Top-K Retriever": {
        "cat": "Retrieval", "in": 1, "out": 1, "priority": 4,
        "stateful": False, "shared_state_allowed": True,
        "supports_branching": False, "supports_loop": False,
        "description": "Direct FAISS retrieval wrapper with configurable k",
        "in_labels": ["query"], "out_labels": ["chunks"],
        "config_options": {"k": 5},
    },
    "Seed Buffer Loader": {
        "cat": "Memory", "in": 0, "out": 1, "priority": 0,
        "stateful": True, "shared_state_allowed": True,
        "supports_branching": False, "supports_loop": False,
        "description": "Pre-loads conversation buffer with seed text at startup",
        "in_labels": [], "out_labels": ["memory"],
        "config_options": {"seed": ""},
    },
}

CANVAS_BOUND = 3000  # half-size

# ─── Sockets ─────────────────────────────────────────────────────────
class SocketItem(QGraphicsItem):
    def __init__(self, node, type_='in', index=0, label="", parent=None):
        super().__init__(parent)
        self.node = node; self.type_ = type_; self.index = index
        self.label = label; self.radius = 9.0
        self.setAcceptHoverEvents(True); self.hovered = False; self.edges = []

    def boundingRect(self):
        return QRectF(-self.radius-60, -self.radius, self.radius*2+120, self.radius*2)

    def paint(self, painter, option, widget):
        r = QRectF(-self.radius, -self.radius, self.radius*2, self.radius*2)
        brush = QBrush(Theme.get("node_sel") if self.hovered else Theme.get("text"))
        painter.setBrush(brush)
        painter.setPen(QPen(Theme.get("node_border"), 2.0))
        painter.drawEllipse(r)
        if self.label:
            painter.setPen(Theme.get("text"))
            painter.setFont(QFont("Segoe UI", 7))
            if self.type_ == 'out':
                painter.drawText(QRectF(self.radius+3, -8, 55, 16), Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter, self.label)
            else:
                painter.drawText(QRectF(-self.radius-58, -8, 55, 16), Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter, self.label)

    def hoverEnterEvent(self, e): self.hovered = True; self.update()
    def hoverLeaveEvent(self, e): self.hovered = False; self.update()
    def get_global_pos(self):
        return self.mapToScene(QPointF(0, 0))


# ─── Edges ───────────────────────────────────────────────────────────
class EdgeItem(QGraphicsPathItem):
    def __init__(self, start_socket, end_socket=None, parent=None):
        super().__init__(parent)
        self.start_socket = start_socket; self.end_socket = end_socket
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.temp_end_pos = None; self.dash_offset = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._pulse)
        self.timer.start(40)

    def _pulse(self):
        if self.end_socket:
            self.dash_offset -= 1.5
            if self.dash_offset < -20: self.dash_offset = 0
            self.update()

    def update_path(self):
        if not self.start_socket: return
        sp = self.start_socket.get_global_pos()
        ep = self.end_socket.get_global_pos() if self.end_socket else self.temp_end_pos
        if ep is None: return
        path = QPainterPath(sp)
        dx = ep.x() - sp.x()
        cx = max(abs(dx)/2, 40)
        path.cubicTo(QPointF(sp.x()+cx, sp.y()), QPointF(ep.x()-cx, ep.y()), ep)
        self.setPath(path)

    def paint(self, painter, option, widget):
        path = self.path()
        bc = Theme.get("node_sel") if self.isSelected() else Theme.get("edge")
        if self.end_socket:
            painter.setPen(QPen(Theme.get("edge_glow"), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(path)
            pen = QPen(bc, 2.5); pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([6, 6]); pen.setDashOffset(self.dash_offset)
            painter.setPen(pen)
        else:
            painter.setPen(QPen(bc, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path)


# ─── Node ────────────────────────────────────────────────────────────
class NodeItem(QGraphicsItem):
    def __init__(self, node_title, node_type, pos_x=0, pos_y=0, scene_ref=None, node_id=None):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        self.scene_ref = scene_ref
        self.node_id = node_id or str(uuid.uuid4())
        self.title = node_title; self.node_type = node_type
        self.spec = NODE_SPEC.get(node_type, {"cat":"Utility","in":1,"out":1,"priority":50,
            "stateful":False,"shared_state_allowed":False,"supports_branching":False,
            "supports_loop":False,"description":"","in_labels":[],"out_labels":[],"config_options":{}})
        self.cat_color = NODE_COLORS.get(self.spec["cat"], Theme.get("node_sel"))

        self.width = 280; self.base_height = 36; self.content_height = 80
        self.collapsed = False; self._is_resizing = False
        self.inputs = []; self.outputs = []
        self.setPos(pos_x, pos_y)

        self.proxy = QGraphicsProxyWidget(self)
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(15, 10, 15, 20)
        self.layout.setSpacing(5)
        self.status_dot = QLabel("\U0001f7e2 Idle")
        self.layout.addWidget(self.status_dot)
        self._build_ui()
        self.apply_theme()

        self.widget.resize(int(self.width), int(self.content_height))
        self.proxy.setWidget(self.widget)
        self.proxy.setPos(0, int(self.base_height))
        self.update_geometry(); self._setup_sockets()

    def apply_theme(self):
        t, w, bb, bh, bt, bd = (Theme.get_str("text"), Theme.get_str("widget_bg"),
            Theme.get_str("btn_bg"), Theme.get_str("btn_hover"),
            Theme.get_str("btn_text"), Theme.get_str("node_border"))
        self.widget.setStyleSheet(f"""
            QWidget{{background:transparent;color:{t};font-family:'Segoe UI';font-size:12px}}
            QTextEdit,QLineEdit,QSpinBox,QComboBox{{background:{w};color:{t};border:1px solid {bd};border-radius:4px;padding:4px}}
            QPushButton{{background:{bb};color:{bt};border:1px solid {bd};border-radius:5px;padding:6px}}
            QPushButton:hover{{background:{bh}}}""")
        self.status_dot.setStyleSheet(f"color:{t};font-weight:bold")
        self.update()

    def _build_ui(self):
        s = self.spec
        if self.node_type == "PDF Loader":
            self.lbl_file = QLabel("No file selected"); self.lbl_file.setWordWrap(True)
            self.btn = QPushButton("Browse Files")
            self.btn.clicked.connect(lambda: self.scene_ref.signals.nodeAction.emit(self.node_id, "upload_pdf", {}))
            self.layout.addWidget(self.btn); self.layout.addWidget(self.lbl_file)
            self.content_height = 90

        elif self.node_type == "FAISS DB":
            row = QHBoxLayout(); row.addWidget(QLabel("k:"))
            self.spin_k = QSpinBox(); self.spin_k.setValue(s["config_options"].get("k",4))
            row.addWidget(self.spin_k); self.layout.addLayout(row)
            self.btn = QPushButton("Index Database")
            self.btn.clicked.connect(lambda: self.scene_ref.signals.nodeAction.emit(self.node_id, "build_db", {}))
            self.layout.addWidget(self.btn)
            self.btn_clear = QPushButton("\U0001f5d1 Clear Index")
            self.btn_clear.clicked.connect(lambda: self.scene_ref.signals.nodeAction.emit(self.node_id, "clear_db", {}))
            self.layout.addWidget(self.btn_clear)
            self.content_height = 140

        elif self.node_type == "Ollama LLM":
            self.layout.addWidget(QLabel("<b>Model:</b> llama3.2:latest"))
            self.content_height = 60

        elif self.node_type == "Query Input":
            self.text_input = QTextEdit(); self.text_input.setPlaceholderText("Enter query...")
            self.text_input.setMinimumHeight(50); self.layout.addWidget(self.text_input)
            self.btn_quick_run = QPushButton("\u25b6 Quick Run (cached)")
            self.btn_quick_run.setStyleSheet("background:#7C3AED;color:white;border:none;font-weight:bold;border-radius:5px;padding:5px")
            self.btn_quick_run.clicked.connect(lambda: self.scene_ref.signals.nodeAction.emit(
                self.node_id, "run_query_node", {"query": self.text_input.toPlainText()}))
            self.layout.addWidget(self.btn_quick_run); self.content_height = 150

        elif self.node_type == "Response Output":
            self.text_output = QTextEdit(); self.text_output.setReadOnly(True)
            self.text_output.setPlaceholderText("Output stream...")
            self.text_output.setMinimumHeight(60)
            self.layout.addWidget(self.text_output); self.content_height = 170

        elif self.node_type == "Buffer":
            row = QHBoxLayout(); row.addWidget(QLabel("Mode:"))
            self.combo_mode = QComboBox(); self.combo_mode.addItems(["shared","independent"])
            row.addWidget(self.combo_mode); self.layout.addLayout(row)
            row2 = QHBoxLayout(); row2.addWidget(QLabel("Window:"))
            self.spin_win = QSpinBox(); self.spin_win.setValue(s["config_options"].get("window_size",5))
            self.spin_win.setMinimum(1); self.spin_win.setMaximum(100)
            row2.addWidget(self.spin_win); self.layout.addLayout(row2)
            self.peek_text = QTextEdit(); self.peek_text.setReadOnly(True)
            self.peek_text.setPlaceholderText("History peek...")
            self.peek_text.setFixedHeight(80); self.peek_text.setFont(QFont("Consolas", 8))
            self.layout.addWidget(self.peek_text)
            self.btn_clear_buf = QPushButton("\U0001f5d1 Clear History")
            self.btn_clear_buf.clicked.connect(lambda: self.scene_ref.signals.nodeAction.emit(self.node_id, "clear_buffer", {}))
            self.layout.addWidget(self.btn_clear_buf)
            self.content_height = 230

        elif self.node_type == "Cache":
            row = QHBoxLayout(); row.addWidget(QLabel("TTL(s):"))
            self.spin_ttl = QSpinBox(); self.spin_ttl.setMaximum(86400)
            self.spin_ttl.setValue(s["config_options"].get("ttl_seconds",3600))
            row.addWidget(self.spin_ttl); self.layout.addLayout(row)
            row2 = QHBoxLayout(); row2.addWidget(QLabel("Max:"))
            self.spin_max = QSpinBox(); self.spin_max.setMaximum(10000)
            self.spin_max.setValue(s["config_options"].get("max_entries",100))
            row2.addWidget(self.spin_max); self.layout.addLayout(row2)
            self.peek_text = QTextEdit(); self.peek_text.setReadOnly(True)
            self.peek_text.setPlaceholderText("Cache peek...")
            self.peek_text.setFixedHeight(80); self.peek_text.setFont(QFont("Consolas", 8))
            self.layout.addWidget(self.peek_text)
            self.btn_clear_cache = QPushButton("\U0001f5d1 Clear Cache")
            self.btn_clear_cache.clicked.connect(lambda: self.scene_ref.signals.nodeAction.emit(self.node_id, "clear_cache", {}))
            self.layout.addWidget(self.btn_clear_cache)
            self.content_height = 230

        elif self.node_type == "Router":
            row = QHBoxLayout(); row.addWidget(QLabel("Type:"))
            self.combo_rt = QComboBox(); self.combo_rt.addItems(["keyword","confidence","manual"])
            row.addWidget(self.combo_rt); self.layout.addLayout(row)
            self.line_kw = QLineEdit(); self.line_kw.setPlaceholderText("keyword filter...")
            self.layout.addWidget(self.line_kw); self.content_height = 100

        elif self.node_type in ["Merge", "Copy", "Multi Merge"]:
            self.layout.addWidget(QLabel(s["description"])); self.content_height = 60

        # ── NEW NODES ────────────────────────────────────────────
        elif self.node_type == "Prompt Template":
            DEFAULT_TEMPLATE = (
                "You are a helpful assistant.\n"
                "Context:\n{context}\n\n"
                "Conversation:\n{memory}\n\n"
                "Question:\n{query}"
            )
            self.tmpl_edit = QTextEdit()
            self.tmpl_edit.setPlaceholderText("Template with {query}, {context}, {memory}...")
            self.tmpl_edit.setText(DEFAULT_TEMPLATE)
            self.tmpl_edit.setMinimumHeight(100)
            self.layout.addWidget(self.tmpl_edit)
            self._default_template = DEFAULT_TEMPLATE
            btn_reset = QPushButton("Reset to Default")
            btn_reset.clicked.connect(lambda: self.tmpl_edit.setText(self._default_template))
            self.layout.addWidget(btn_reset)
            self.content_height = 190

        elif self.node_type == "System Message":
            self.sys_edit = QTextEdit()
            self.sys_edit.setPlaceholderText("Enter system instruction...")
            self.sys_edit.setText("You are a knowledgeable, concise, and helpful AI assistant.")
            self.sys_edit.setMinimumHeight(70)
            self.layout.addWidget(self.sys_edit)
            self.content_height = 120

        elif self.node_type == "Reranker":
            row = QHBoxLayout(); row.addWidget(QLabel("Top-K:"))
            self.spin_topk = QSpinBox()
            self.spin_topk.setMinimum(1); self.spin_topk.setMaximum(20)
            self.spin_topk.setValue(s["config_options"].get("top_k", 3))
            row.addWidget(self.spin_topk); self.layout.addLayout(row)
            self.layout.addWidget(QLabel("(Sorts chunks by score, keeps top-K)"))
            self.content_height = 90

        elif self.node_type == "Memory Formatter":
            row = QHBoxLayout(); row.addWidget(QLabel("Max tokens:"))
            self.spin_maxtok = QSpinBox()
            self.spin_maxtok.setMinimum(50); self.spin_maxtok.setMaximum(4000)
            self.spin_maxtok.setValue(s["config_options"].get("max_tokens", 500))
            self.spin_maxtok.setSingleStep(50)
            row.addWidget(self.spin_maxtok); self.layout.addLayout(row)
            self.cb_summarize = QCheckBox("Summarize (truncate only if off)")
            self.layout.addWidget(self.cb_summarize)
            self.content_height = 100

        elif self.node_type == "Conversation Starter":
            self.starter_edit = QTextEdit()
            self.starter_edit.setPlaceholderText("Initial context or greeting...")
            self.starter_edit.setMinimumHeight(60)
            self.layout.addWidget(self.starter_edit)
            self.content_height = 120

        elif self.node_type == "Score Filter":
            row = QHBoxLayout(); row.addWidget(QLabel("Min Score:"))
            self.spin_thresh = QDoubleSpinBox()
            self.spin_thresh.setRange(0.0, 1.0)
            self.spin_thresh.setSingleStep(0.05)
            self.spin_thresh.setDecimals(2)
            self.spin_thresh.setValue(s["config_options"].get("threshold", 0.5))
            row.addWidget(self.spin_thresh); self.layout.addLayout(row)
            self.layout.addWidget(QLabel("Drops chunks below threshold"))
            self.content_height = 90

        elif self.node_type == "Debug Inspector":
            self.debug_out = QTextEdit()
            self.debug_out.setReadOnly(True)
            self.debug_out.setPlaceholderText("Data will appear here during execution...")
            self.debug_out.setMinimumHeight(80)
            self.layout.addWidget(self.debug_out)
            self.content_height = 140

        elif self.node_type == "Top-K Retriever":
            row = QHBoxLayout(); row.addWidget(QLabel("k:"))
            self.spin_topk_ret = QSpinBox()
            self.spin_topk_ret.setMinimum(1); self.spin_topk_ret.setMaximum(50)
            self.spin_topk_ret.setValue(s["config_options"].get("k", 5))
            row.addWidget(self.spin_topk_ret); self.layout.addLayout(row)
            self.layout.addWidget(QLabel("Direct FAISS similarity_search"))
            self.content_height = 90

        elif self.node_type == "Seed Buffer Loader":
            self.seed_edit = QTextEdit()
            self.seed_edit.setPlaceholderText("Seed conversation turns (one per line)...")
            self.seed_edit.setMinimumHeight(70)
            self.layout.addWidget(self.seed_edit)
            btn_inject = QPushButton("Inject Seed Into Buffer")
            btn_inject.clicked.connect(lambda: self.scene_ref.signals.nodeAction.emit(
                self.node_id, "inject_seed", {"seed": self.seed_edit.toPlainText()}))
            self.layout.addWidget(btn_inject)
            self.content_height = 160

    # ── data access ──
    def set_data(self, key, val):
        if key == "status": self.status_dot.setText(val)
        elif key == "peek":
            if hasattr(self, 'peek_text'):
                self.peek_text.setPlainText(str(val))
        
        if self.node_type == "PDF Loader" and key == "file": self.lbl_file.setText(val)
        elif self.node_type == "Response Output" and key == "answer": self.text_output.setText(val)
        elif self.node_type == "Debug Inspector" and key == "data":
            if hasattr(self, 'debug_out'):
                self.debug_out.setPlainText(str(val)[:2000])

    def get_data(self, key):
        if self.node_type == "Query Input" and key == "query":
            return self.text_input.toPlainText()
        if self.node_type == "Prompt Template" and key == "template":
            return self.tmpl_edit.toPlainText() if hasattr(self, 'tmpl_edit') else ""
        if self.node_type == "System Message" and key == "message":
            return self.sys_edit.toPlainText() if hasattr(self, 'sys_edit') else ""
        if self.node_type == "Reranker" and key == "top_k":
            return self.spin_topk.value() if hasattr(self, 'spin_topk') else 3
        if self.node_type == "Memory Formatter" and key == "max_tokens":
            return self.spin_maxtok.value() if hasattr(self, 'spin_maxtok') else 500
        if self.node_type == "Memory Formatter" and key == "summarize":
            return self.cb_summarize.isChecked() if hasattr(self, 'cb_summarize') else False
        if self.node_type == "Conversation Starter" and key == "text":
            return self.starter_edit.toPlainText() if hasattr(self, 'starter_edit') else ""
        if self.node_type == "Score Filter" and key == "threshold":
            return self.spin_thresh.value() if hasattr(self, 'spin_thresh') else 0.5
        if self.node_type == "Top-K Retriever" and key == "k":
            return self.spin_topk_ret.value() if hasattr(self, 'spin_topk_ret') else 5
        if self.node_type == "Seed Buffer Loader" and key == "seed":
            return self.seed_edit.toPlainText() if hasattr(self, 'seed_edit') else ""
        return None

    # ── geometry ──
    def update_geometry(self):
        self.height = self.base_height if self.collapsed else self.base_height + self.content_height

    def _setup_sockets(self):
        s = self.spec
        in_labels = s.get("in_labels", [])
        out_labels = s.get("out_labels", [])
        n_in, n_out = s["in"], s["out"]
        isp = self.height / (n_in + 1) if n_in > 0 else 0
        osp = self.height / (n_out + 1) if n_out > 0 else 0
        for i in range(n_in):
            lbl = in_labels[i] if i < len(in_labels) else ""
            sock = SocketItem(self, 'in', i, lbl, self)
            sock.setPos(0, isp * (i + 1)); self.inputs.append(sock)
        for i in range(n_out):
            lbl = out_labels[i] if i < len(out_labels) else ""
            sock = SocketItem(self, 'out', i, lbl, self)
            sock.setPos(self.width, osp * (i + 1)); self.outputs.append(sock)

    def reposition_sockets(self):
        n_in, n_out = self.spec["in"], self.spec["out"]
        isp = self.height / (n_in + 1) if n_in > 0 else 0
        osp = self.height / (n_out + 1) if n_out > 0 else 0
        for i, s in enumerate(self.inputs): s.setPos(0, isp*(i+1))
        for i, s in enumerate(self.outputs): s.setPos(self.width, osp*(i+1))
        for s in self.inputs + self.outputs:
            for e in s.edges: e.update_path()

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def is_resize_zone(self, pos):
        return pos.x() > self.width - 20 and pos.y() > self.height - 20 and not self.collapsed

    def hoverMoveEvent(self, ev):
        self.setCursor(Qt.CursorShape.SizeFDiagCursor if self.is_resize_zone(ev.pos()) else Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(ev)

    def mousePressEvent(self, ev):
        if self.is_resize_zone(ev.pos()): self._is_resizing = True; return
        if ev.pos().y() <= self.base_height and ev.pos().x() >= self.width - 25:
            self.scene_ref.delete_node(self); return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._is_resizing:
            self.prepareGeometryChange()
            self.width = max(220, ev.pos().x())
            self.content_height = max(60, ev.pos().y() - self.base_height)
            self.widget.setFixedSize(int(self.width), int(self.content_height))
            self.proxy.resize(self.width, self.content_height)
            self.update_geometry(); self.reposition_sockets(); self.update(); return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._is_resizing: self._is_resizing = False; return
        super().mouseReleaseEvent(ev)

    def toggle_collapse(self):
        self.collapsed = not self.collapsed
        self.proxy.setVisible(not self.collapsed)
        self.prepareGeometryChange(); self.update_geometry()
        self.reposition_sockets(); self.update()

    def mouseDoubleClickEvent(self, ev):
        if ev.pos().y() <= self.base_height: self.toggle_collapse()
        else: super().mouseDoubleClickEvent(ev)

    def paint(self, painter, option, widget):
        rect = self.boundingRect()
        s = self.spec
        # shadow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Theme.get("shadow"))
        painter.drawRoundedRect(rect.translated(0, 8), 12, 12)
        # body
        painter.setPen(QPen(Theme.get("node_sel") if self.isSelected() else Theme.get("node_border"), 1.5))
        painter.setBrush(Theme.get("node_bg"))
        painter.drawRoundedRect(rect, 10, 10)
        # header
        hp = QPainterPath(); hp.addRoundedRect(0, 0, self.width, self.base_height, 10, 10)
        hp.addRect(0, self.base_height - 10, self.width, 10)
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(self.cat_color)
        painter.drawPath(hp.simplified())
        # header text color
        hc = QColor("#11111B") if Theme.current == "dark" else QColor("#FFFFFF")
        # collapse icon
        painter.setPen(hc); painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(QRectF(8, 0, 24, self.base_height), Qt.AlignmentFlag.AlignCenter, "+" if self.collapsed else "\u2013")
        # title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(30, 0, self.width - 110, self.base_height), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.title)
        # priority badge
        p_text = f"P:{s['priority']}"
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        pw = painter.fontMetrics().horizontalAdvance(p_text) + 10
        pr = QRectF(self.width - 60 - pw, 6, pw, 22)
        painter.setBrush(QColor(0, 0, 0, 60)); painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(pr, 6, 6)
        painter.setPen(hc); painter.drawText(pr, Qt.AlignmentFlag.AlignCenter, p_text)
        # stateful icon
        if s.get("stateful"):
            painter.setBrush(QColor("#F9E2AF")); painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(self.width - 50, self.base_height / 2), 5, 5)
        # delete X
        painter.setPen(QColor("#EF4444") if self.isSelected() else hc)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(QRectF(self.width - 25, 0, 20, self.base_height), Qt.AlignmentFlag.AlignCenter, "X")
        # resize grip
        if not self.collapsed:
            painter.setPen(QPen(Theme.get("node_border"), 2))
            painter.drawLine(QPointF(self.width-6, self.height-14), QPointF(self.width-14, self.height-6))
            painter.drawLine(QPointF(self.width-6, self.height-9), QPointF(self.width-9, self.height-6))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            b = CANVAS_BOUND
            h = getattr(self, 'height', 200)
            x = max(-b, min(value.x(), b - self.width))
            y = max(-b, min(value.y(), b - h))
            return QPointF(x, y)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for s in self.inputs + self.outputs:
                for e in s.edges: e.update_path()
        return super().itemChange(change, value)


# ─── Undo Commands ───────────────────────────────────────────────────
class AddNodeCmd(QUndoCommand):
    def __init__(self, scene, ntype, pos, node_id=None):
        super().__init__(f"Add {ntype}")
        self.scene = scene; self.ntype = ntype; self.pos = pos
        self.nid = node_id or str(uuid.uuid4()); self.first = True

    def redo(self):
        if self.first: self.first = False; return
        self.scene._raw_add_node(self.ntype, self.pos, self.nid)

    def undo(self):
        node = self.scene._find_node(self.nid)
        if node: self.scene._raw_delete_node(node)


class DeleteNodeCmd(QUndoCommand):
    def __init__(self, scene, node):
        super().__init__(f"Delete {node.node_type}")
        self.scene = scene; self.ntype = node.node_type
        self.nid = node.node_id; self.pos = node.pos()

    def redo(self):
        node = self.scene._find_node(self.nid)
        if node: self.scene._raw_delete_node(node)

    def undo(self):
        self.scene._raw_add_node(self.ntype, self.pos, self.nid)


# ─── Scene ───────────────────────────────────────────────────────────
class SceneSignals(QObject):
    nodeAdded = pyqtSignal(object)
    nodeDeleted = pyqtSignal(str)
    graphEdgesChanged = pyqtSignal()
    nodeAction = pyqtSignal(str, str, dict)


class NodeScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-CANVAS_BOUND, -CANVAS_BOUND, CANVAS_BOUND*2, CANVAS_BOUND*2)
        self.setBackgroundBrush(Theme.get("bg"))
        self.drag_edge = None
        self.signals = SceneSignals()
        self.undo_stack = QUndoStack()

    def update_theme(self):
        self.setBackgroundBrush(Theme.get("bg"))
        for item in self.items():
            if isinstance(item, NodeItem): item.apply_theme()
            elif isinstance(item, EdgeItem): item.update()
        self.update()

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        grid = 30
        l = int(rect.left()//grid)*grid; r = int(rect.right()//grid)*grid
        t = int(rect.top()//grid)*grid; b = int(rect.bottom()//grid)*grid
        pts = [QPointF(x, y) for x in range(l, r, grid) for y in range(t, b, grid)]
        painter.setPen(QPen(Theme.get("grid"), 2)); painter.drawPoints(pts)
        # boundary
        br = QRectF(-CANVAS_BOUND, -CANVAS_BOUND, CANVAS_BOUND*2, CANVAS_BOUND*2)
        painter.setPen(QPen(Theme.get("boundary"), 3, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(br)

    def mousePressEvent(self, event):
        item = self.itemAt(event.scenePos(), QGraphicsView().transform())
        if isinstance(item, SocketItem) and item.type_ == 'out':
            self.drag_edge = EdgeItem(item, None)
            self.drag_edge.temp_end_pos = event.scenePos()
            self.addItem(self.drag_edge); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_edge:
            self.drag_edge.temp_end_pos = event.scenePos()
            self.drag_edge.update_path(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_edge:
            item = self.itemAt(event.scenePos(), QGraphicsView().transform())
            if isinstance(item, SocketItem) and item.type_ == 'in' and item.node != self.drag_edge.start_socket.node:
                self.drag_edge.end_socket = item
                self.drag_edge.start_socket.edges.append(self.drag_edge)
                item.edges.append(self.drag_edge)
                self.drag_edge.update_path()
                self.signals.graphEdgesChanged.emit()
            else:
                self.removeItem(self.drag_edge)
            self.drag_edge = None; return
        super().mouseReleaseEvent(event)

    # raw (no undo) helpers
    def _raw_add_node(self, ntype, pos, nid=None):
        node = NodeItem(ntype, ntype, pos.x(), pos.y(), self, nid)
        self.addItem(node); self.signals.nodeAdded.emit(node); return node

    def _raw_delete_node(self, node):
        nid = node.node_id
        for sock in node.inputs + node.outputs:
            for e in list(sock.edges):
                if e.start_socket: e.start_socket.edges.remove(e)
                if e.end_socket: e.end_socket.edges.remove(e)
                self.removeItem(e)
        self.removeItem(node); self.signals.nodeDeleted.emit(nid)

    def _find_node(self, nid):
        for i in self.items():
            if isinstance(i, NodeItem) and i.node_id == nid: return i
        return None

    # public (undo-tracked)
    def add_node(self, ntype, pos, nid=None):
        nid = nid or str(uuid.uuid4())
        cmd = AddNodeCmd(self, ntype, pos, nid)
        node = self._raw_add_node(ntype, pos, nid)
        self.undo_stack.push(cmd)
        return node

    def delete_node(self, node):
        cmd = DeleteNodeCmd(self, node)
        self.undo_stack.push(cmd)

    def delete_selected(self):
        for item in list(self.selectedItems()):
            if isinstance(item, EdgeItem):
                if item.start_socket: item.start_socket.edges.remove(item)
                if item.end_socket: item.end_socket.edges.remove(item)
                self.removeItem(item)
        for item in list(self.selectedItems()):
            if isinstance(item, NodeItem): self.delete_node(item)
        self.signals.graphEdgesChanged.emit()

    def serialize(self):
        nodes, edges = [], []
        for i in self.items():
            if isinstance(i, NodeItem):
                nodes.append({"id": i.node_id, "type": i.node_type, "x": i.x(), "y": i.y()})
                for sock in i.outputs:
                    for e in sock.edges:
                        if e.end_socket:
                            edges.append({"from": i.node_id, "fp": sock.index, "to": e.end_socket.node.node_id, "tp": e.end_socket.index})
        return {"nodes": nodes, "edges": edges}

    def deserialize(self, data):
        self.clear(); self.undo_stack.clear()
        nm = {}
        for n in data.get("nodes", []):
            nm[n["id"]] = self._raw_add_node(n["type"], QPointF(n["x"], n["y"]), n["id"])
        for e in data.get("edges", []):
            try:
                n1, n2 = nm[e["from"]], nm[e["to"]]
                s1, s2 = n1.outputs[e["fp"]], n2.inputs[e["tp"]]
                edge = EdgeItem(s1, s2)
                s1.edges.append(edge); s2.edges.append(edge)
                self.addItem(edge); edge.update_path()
            except: pass


# ─── View ────────────────────────────────────────────────────────────
class NodeView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        f = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(f, f)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(f"QMenu{{background:{Theme.get_str('node_bg')};color:{Theme.get_str('text')};border:1px solid {Theme.get_str('node_border')};padding:5px}}QMenu::item:selected{{background:{Theme.get_str('node_sel')};color:#FFF}}")
        for k in NODE_SPEC.keys():
            menu.addAction(f"Add {k}").triggered.connect(lambda c, t=k: self.scene().add_node(t, self.mapToScene(event.pos())))
        menu.exec(event.globalPos())

    def keyPressEvent(self, event):
        # Don't intercept ANY keys when a proxy widget (text field) has focus
        focus_item = self.scene().focusItem()
        if isinstance(focus_item, QGraphicsProxyWidget):
            super().keyPressEvent(event); return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.scene().delete_selected()
        elif event.matches(QKeySequence.StandardKey.Undo):
            self.scene().undo_stack.undo()
        elif event.matches(QKeySequence.StandardKey.Redo):
            self.scene().undo_stack.redo()
        else:
            super().keyPressEvent(event)
