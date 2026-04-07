import sys, os, json
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSplitter, QTextEdit,
    QListWidget, QListWidgetItem, QDialog, QTabWidget,
    QStackedWidget, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QSize
from PyQt6.QtGui import QColor, QKeySequence, QShortcut

from utils import setup_directories
from ingest import DocIngestor
from rag_engine import RAGEngine
from graph_executor import GraphExecutor
from node_editor import (NodeScene, NodeView, NodeItem, EdgeItem,
                         Theme, NODE_SPEC)
from examples_tab import ExamplesTab
from sessions import SessionManager


class IngestThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    def __init__(self, pdf_paths):
        super().__init__(); self.pdf_paths = pdf_paths
    def run(self):
        try:
            self.finished.emit(DocIngestor().ingest_pdfs(self.pdf_paths))
        except Exception as e: self.error.emit(str(e))


class GraphRunThread(QThread):
    """Runs GraphExecutor in a background thread."""
    finished = pyqtSignal(str)   # final output text
    error = pyqtSignal(str)
    log_signal = pyqtSignal(str, str)      # text, color
    status_signal = pyqtSignal(str, str)   # node_id, status

    def __init__(self, graph_data, engine, node_configs, reset_buffer=False):
        super().__init__()
        self.graph_data = graph_data
        self.engine = engine
        self.node_configs = node_configs
        self.reset_buffer = reset_buffer
        self.query_text = ""

    def run(self):
        try:
            ex = GraphExecutor(
                self.graph_data, self.engine,
                log_fn=lambda t, c="#89B4FA": self.log_signal.emit(t, c),
                status_fn=lambda nid, s: self.status_signal.emit(nid, s),
                reset_buffer=self.reset_buffer
            )
            for nid, cfg in self.node_configs.items():
                ex.set_node_config(nid, cfg)
            result = ex.execute(self.query_text)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ─── Architecture Diagram Generator ─────────────────────────────────
def generate_arch_diagram(scene):
    data = scene.serialize()
    nodes = {n["id"]: n for n in data["nodes"]}
    adj = {}  # id -> [(target_id, out_port_idx)]
    for e in data["edges"]:
        adj.setdefault(e["from"], []).append((e["to"], e["fp"]))
    # find roots (no incoming)
    has_incoming = {e["to"] for e in data["edges"]}
    roots = [n["id"] for n in data["nodes"] if n["id"] not in has_incoming]
    if not roots:
        roots = [data["nodes"][0]["id"]] if data["nodes"] else []

    lines = ["Architecture Diagram", "=" * 40, ""]
    visited = set()

    def walk(nid, indent=0, port_label=""):
        if nid in visited:
            spec = NODE_SPEC.get(nodes[nid]["type"], {})
            prefix = " " * indent
            lines.append(f"{prefix}(cycle back to) [{nodes[nid]['type']}]")
            return
        visited.add(nid)
        spec = NODE_SPEC.get(nodes[nid]["type"], {})
        p = spec.get("priority", "?")
        prefix = " " * indent
        port_str = f"({port_label}) " if port_label else ""
        badge = "\u25cf" if spec.get("stateful") else ""
        lines.append(f"{prefix}{port_str}[P:{p}] {nodes[nid]['type']} {badge}")
        children = adj.get(nid, [])
        for i, (child_id, out_port) in enumerate(children):
            out_labels = spec.get("out_labels", [])
            ol = out_labels[out_port] if out_port < len(out_labels) else ""
            connector = "\u2514\u2500\u2500\u25b6 " if i == len(children)-1 else "\u251c\u2500\u2500\u25b6 "
            lines.append(f"{prefix}  {connector}")
            walk(child_id, indent + 6, ol)

    for r in roots:
        walk(r)
        lines.append("")

    return "\n".join(lines)


class ArchDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Architecture Diagram")
        self.resize(650, 500)
        layout = QVBoxLayout(self)
        self.te = QTextEdit()
        self.te.setReadOnly(True)
        self.te.setFont(self.te.font())
        self.te.setPlainText(text)
        self.te.setStyleSheet(f"background:{Theme.get_str('widget_bg')};color:{Theme.get_str('text')};font-family:Consolas;font-size:13px;border:none;padding:12px")
        layout.addWidget(self.te)
        btn = QPushButton("Copy to Clipboard")
        btn.clicked.connect(lambda: QApplication.clipboard().setText(text))
        layout.addWidget(btn)
        self.setStyleSheet(f"background:{Theme.get_str('bg')};color:{Theme.get_str('text')}")


class SessionItemWidget(QWidget):
    """Custom widget for QListWidgetItem with expandable chat history and Alias support."""
    deleteRequested = pyqtSignal(str)
    sizeChanged = pyqtSignal(int)
    aliasRequested = pyqtSignal(str)

    def __init__(self, name, session_id, alias="", buffer=None, parent=None):
        super().__init__(parent)
        self.session_id = session_id
        self.is_expanded = False
        self.buffer = buffer or []
        self.name = name
        self.alias = alias
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Top Row
        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        self.top_layout.setContentsMargins(8, 4, 8, 4)
        
        self.btn_toggle = QPushButton("▶") # More standard bold arrow
        self.btn_toggle.setFixedSize(24, 24)
        self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle.clicked.connect(self.toggle_expansion)
        self.top_layout.addWidget(self.btn_toggle)
        
        self.label = QLabel()
        self.refresh_display()
        self.top_layout.addWidget(self.label)
        self.top_layout.addStretch()
        
        self.btn_del = QPushButton("✖") # Bold X icon
        self.btn_del.setFixedSize(24, 24)
        self.btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_del.clicked.connect(lambda: self.deleteRequested.emit(self.session_id))
        self.top_layout.addWidget(self.btn_del)
        
        self.main_layout.addWidget(self.top_widget)
        
        # Expanded History View
        self.history_view = QTextEdit()
        self.history_view.setReadOnly(True)
        self.history_view.setVisible(False)
        self.history_view.setFixedHeight(120)
        self.history_view.setPlainText("\n\n".join(self.buffer) if self.buffer else "No history.")
        self.main_layout.addWidget(self.history_view)
        
        self.apply_widget_theme()

    def refresh_display(self):
        txt = self.alias if self.alias else self.name
        self.label.setText(txt)
        if self.alias:
            self.label.setToolTip(f"Internal: {self.name}")
        else:
            self.label.setToolTip("")

    def apply_widget_theme(self):
        txt_col = "#CDD6F4" # Explicit high-contrast light
        btn_hov = "#89B4FA" # Subtle blue highlight on hover
        
        # High-contrast controls: Absolute white, no padding, bold symbolic font
        # We add a slight background color to ensure they are visible even when not hovered
        btn_style = (
            f"QPushButton {{ "
            f"  background: #313244; "
            f"  color: #FFFFFF !important; "
            f"  border: 1px solid #45475A; "
            f"  border-radius: 4px; "
            f"  font-weight: 900; "
            f"  font-size: 16px; "
            f"  padding: 0px; "
            f"  margin: 0px; "
            f"  font-family: 'Segoe UI Symbol', 'Arial Black', sans-serif;"
            f"  outline: none;"
            f"}}"
            f"QPushButton:hover {{ "
            f"  background: {btn_hov}99; "
            f"  border: 1px solid white; "
            f"  color: white !important;"
            f"}}"
        )
        self.btn_toggle.setStyleSheet(btn_style)
        self.btn_del.setStyleSheet(btn_style)
        self.label.setStyleSheet(f"color:{txt_col}; font-family:'Segoe UI'; font-size:12px; border:none;")
        self.history_view.setStyleSheet(
            f"background:#1E1E2E; color:{txt_col}; border:none; "
            f"border-top:1px solid #45475A; font-family:'Cascadia Code'; font-size:11px; padding:8px;"
        )

    def toggle_expansion(self):
        self.is_expanded = not self.is_expanded
        self.history_view.setVisible(self.is_expanded)
        self.btn_toggle.setText("▼" if self.is_expanded else "▶")
        new_h = 160 if self.is_expanded else 40
        self.sizeChanged.emit(new_h)

# ─── Main Window ─────────────────────────────────────────────────────
class RAGWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAG Node Orchestrator")
        self.resize(1400, 850)
        setup_directories(os.getcwd())
        self.engine = RAGEngine()
        self.node_scene = NodeScene()
        self.node_scene.signals.nodeAction.connect(self.handle_node_action)
        self.node_scene.signals.nodeAdded.connect(self.on_node_added)
        self.node_scene.signals.nodeDeleted.connect(self.on_node_deleted)
        self.session_manager = SessionManager()
        self.init_ui()
        self.refresh_session_list()
        self.try_load_db()
        self.build_default_architecture()
        self.apply_global_theme()

    def log(self, text, color="#89B4FA"):
        if Theme.current == "light":
            color = {"#CDD6F4":"#1F2937","#89B4FA":"#3B82F6","#F38BA8":"#EF4444","#A6E3A1":"#059669","#F9E2AF":"#92400E","#CBA6F7":"#7C3AED"}.get(color, "#1F2937")
        self.terminal.append(f"<span style='color:{color}'>&gt;&gt; {text}</span>")

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        ml = QVBoxLayout(self.central_widget)
        ml.setContentsMargins(0,0,0,0); ml.setSpacing(0)

        # toolbar
        tb = QWidget(); tb.setStyleSheet("background:transparent")
        tbl = QHBoxLayout(tb); tbl.setContentsMargins(15,8,15,8)
        for txt, fn in [
            ("Save", self.save_architecture),
            ("Load", self.load_architecture),
            ("Clear", lambda: self.node_scene.clear()),
            ("\u21a9 Undo", lambda: self.node_scene.undo_stack.undo()),
            ("\u21aa Redo", lambda: self.node_scene.node_scene.undo_stack.redo() if hasattr(self.node_scene, 'node_scene') else self.node_scene.undo_stack.redo()),
            ("\U0001f4d0 View Arch", self.show_arch_diagram),
            ("History", self.toggle_history_sidebar),
        ]:
            b = QPushButton(txt); b.clicked.connect(fn); tbl.addWidget(b)
        tbl.addStretch()
        ml.addWidget(tb)

        # main h-split
        self.h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT sidebar
        sw = QWidget()
        sl = QVBoxLayout(sw); sl.setContentsMargins(12,12,12,12)
        sl.addWidget(QLabel("<b>Pipeline Orchestrator</b>"))
        self.btn_run = QPushButton("\U0001f680 Full Pipeline Run")
        self.btn_run.setStyleSheet("background:#10B981;color:white;border:none;font-weight:bold;border-radius:5px;padding:12px")
        self.btn_run.clicked.connect(self.run_global_pipeline)
        sl.addWidget(self.btn_run)
        btn_row = QHBoxLayout()
        ba = QPushButton("\U0001f504 Auto-Sort"); ba.clicked.connect(self.auto_sort_topology)
        bl = QPushButton("\u2795 Add Level"); bl.clicked.connect(self.add_manual_rank)
        btn_row.addWidget(ba); btn_row.addWidget(bl)
        sl.addLayout(btn_row)
        self.rank_list = QListWidget()
        self.rank_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.rank_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        sl.addWidget(self.rank_list)
        self.h_splitter.addWidget(sw)

        # RIGHT (canvas + terminal)
        rp = QWidget(); rl = QVBoxLayout(rp); rl.setContentsMargins(0,0,0,0); rl.setSpacing(0)
        self.v_splitter = QSplitter(Qt.Orientation.Vertical)
        self.node_view = NodeView(self.node_scene)
        self.node_view.setStyleSheet("border:none;background:transparent")
        self.v_splitter.addWidget(self.node_view)
        self.terminal = QTextEdit(); self.terminal.setReadOnly(True)
        self.terminal.setText("<span>&gt;&gt; RAG Orchestration OS Online.</span>")
        self.v_splitter.addWidget(self.terminal)
        self.v_splitter.setSizes([650, 200])
        rl.addWidget(self.v_splitter)
        self.h_splitter.addWidget(rp)

        # RIGHT sidebar (Chat History)
        self.history_sidebar = QWidget()
        self.history_layout = QVBoxLayout(self.history_sidebar)
        self.history_layout.setContentsMargins(12,12,12,12)
        
        # Header
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<b>Chat History</b>"))
        header_row.addStretch()
        self.history_layout.addLayout(header_row)

        self.btn_new_chat = QPushButton("\u2795 New Chat")
        self.btn_new_chat.setStyleSheet("background:#7C3AED;color:white;border:none;font-weight:bold;border-radius:5px;padding:10px")
        self.btn_new_chat.clicked.connect(self.create_new_session)
        self.history_layout.addWidget(self.btn_new_chat)
        
        self.session_list = QListWidget()
        self.session_list.itemClicked.connect(self.on_session_clicked)
        self.session_list.itemDoubleClicked.connect(self.set_session_alias_item) # Focus on Alias
        self.session_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.session_list.customContextMenuRequested.connect(self.show_session_context_menu)
        self.history_layout.addWidget(self.session_list)
        
        self.h_splitter.addWidget(self.history_sidebar)
        self.h_splitter.setSizes([260, 880, 260])

        # ─── Navigation & Stacked View ──────────────────────────────
        self.main_stack = QStackedWidget()
        
        # Add Nav buttons to LEFT sidebar (sw)
        nav_label = QLabel("<b>Navigation</b>")
        nav_label.setStyleSheet("margin-top:10px; color:#89B4FA")
        sl.insertWidget(0, nav_label)
        
        self.nav_group = QButtonGroup(self)
        self.btn_nav_builder = QPushButton("\U0001f3d7  Graph Builder")
        self.btn_nav_gallery = QPushButton("\U0001f4da  Example Gallery")
        
        for i, b in enumerate([self.btn_nav_builder, self.btn_nav_gallery]):
            b.setCheckable(True)
            b.setFixedHeight(40)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav_group.addButton(b, i)
            sl.insertWidget(i+1, b)
        
        self.btn_nav_builder.setChecked(True)
        self.btn_nav_builder.clicked.connect(lambda: self.main_stack.setCurrentIndex(0))
        self.btn_nav_gallery.clicked.connect(lambda: self.main_stack.setCurrentIndex(1))

        sl.insertSpacing(3, 15) # Space between Nav and Orchestrator
        
        # Add views to stack
        stack_builder = QWidget()
        sbl = QVBoxLayout(stack_builder)
        sbl.setContentsMargins(0,0,0,0); sbl.setSpacing(0)
        sbl.addWidget(self.h_splitter)
        self.main_stack.addWidget(stack_builder)
        
        self.examples_tab = ExamplesTab(
            load_callback=self.load_from_example,
            back_callback=lambda: (self.btn_nav_builder.setChecked(True), self.main_stack.setCurrentIndex(0))
        )
        self.main_stack.addWidget(self.examples_tab)
        
        ml.addWidget(self.main_stack)
        self.apply_global_theme()

    # ── sidebar ──
    def on_node_added(self, node):
        item = QListWidgetItem(f"  \u21b3 {node.node_type} [{node.node_id[:4]}]")
        item.setData(Qt.ItemDataRole.UserRole, node.node_id)
        self.rank_list.addItem(item)

    def on_node_deleted(self, nid):
        for i in range(self.rank_list.count()):
            if self.rank_list.item(i).data(Qt.ItemDataRole.UserRole) == nid:
                self.rank_list.takeItem(i); break

    def add_manual_rank(self):
        cnt = sum(1 for i in range(self.rank_list.count()) if str(self.rank_list.item(i).text()).startswith("==="))
        item = QListWidgetItem(f"=== Level {cnt+1} ===")
        item.setBackground(QColor(Theme.get_str("btn_bg")))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
        self.rank_list.addItem(item)

    def auto_sort_topology(self):
        self.rank_list.clear()
        data = self.node_scene.serialize()
        nodes = data["nodes"]; edges = data["edges"]
        if not nodes: return
        in_deg = {n["id"]: 0 for n in nodes}
        adj = {n["id"]: [] for n in nodes}
        nmap = {n["id"]: n for n in nodes}
        for e in edges:
            adj[e["from"]].append(e["to"]); in_deg[e["to"]] += 1
        queue = sorted([nid for nid, d in in_deg.items() if d == 0],
                       key=lambda nid: NODE_SPEC.get(nmap[nid]["type"], {}).get("priority", 50))
        rank = 1
        while queue:
            ri = QListWidgetItem(f"=== Level {rank} ===")
            ri.setBackground(QColor(Theme.get_str("btn_bg")))
            ri.setFlags(ri.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            self.rank_list.addItem(ri)
            nxt = []
            for nid in queue:
                p = NODE_SPEC.get(nmap[nid]["type"], {}).get("priority", "?")
                item = QListWidgetItem(f"  \u21b3 P:{p} {nmap[nid]['type']} [{nid[:4]}]")
                item.setData(Qt.ItemDataRole.UserRole, nid)
                self.rank_list.addItem(item)
                for nb in adj[nid]:
                    in_deg[nb] -= 1
                    if in_deg[nb] == 0: nxt.append(nb)
            queue = sorted(nxt, key=lambda nid: NODE_SPEC.get(nmap[nid]["type"], {}).get("priority", 50))
            rank += 1
        cycles = [nmap[nid]["type"] for nid, d in in_deg.items() if d > 0]
        if cycles:
            self.log(f"Cycle detected: {', '.join(cycles)}", "#F38BA8")

    # ── theme ──
    def apply_global_theme(self):
        bg, txt, wbg = Theme.get_str('bg'), Theme.get_str('text'), Theme.get_str('widget_bg')
        bord, bbg, bhov = Theme.get_str('node_border'), Theme.get_str('btn_bg'), Theme.get_str('btn_hover')
        
        self.central_widget.setStyleSheet(f"background:{bg};color:{txt}")
        self.terminal.setStyleSheet(f"background:{wbg};color:{txt};font-family:Consolas;padding:10px;border:none;border-top:2px solid {bord}")
        self.h_splitter.widget(0).setStyleSheet(f"background:{wbg};border-right:2px solid {bord}")
        self.history_sidebar.setStyleSheet(f"background:{wbg};border-left:2px solid {bord}")
        self.session_list.setStyleSheet(f"background:transparent;border:none;font-family:'Segoe UI';font-size:13px;color:{txt}")
        
        # Comprehensive widget styling for mid-tone themes
        input_style = (
            f"background:{wbg}; color:{txt}; border:1px solid {bord}; border-radius:4px; padding:4px;"
        )
        self.setStyleSheet(
            f"QSplitter::handle{{background:{bord};margin:2px}}"
            f"QSplitter::handle:horizontal{{width:6px;border-radius:3px}}"
            f"QSplitter::handle:vertical{{height:6px;border-radius:3px}}"
            f"QPushButton{{background:{bbg};color:{txt};border:1px solid {bord};border-radius:6px;padding:6px 12px;font-weight:bold}}"
            f"QPushButton:hover{{background:{bhov}}}"
            f"QPushButton:checked{{background:#89B4FA; color:#1E1E2E}}" # Highlight active nav
            f"QLineEdit{{{input_style}}} QSpinBox{{{input_style}}} QDoubleSpinBox{{{input_style}}} QComboBox{{{input_style}}}"
            f"QTextEdit{{{input_style}}}"
            f"QMenu{{background:{wbg}; color:{txt}; border:1px solid {bord};}}"
            f"QMenu::item:selected{{background:{bhov};}}"
        )
        # Custom styles for main colored buttons
        self.btn_run.setStyleSheet("background:#10B981;color:white;border:none;font-weight:bold;border-radius:5px;padding:12px")
        self.btn_new_chat.setStyleSheet("background:#7C3AED;color:white;border:none;font-weight:bold;border-radius:5px;padding:10px")

        # Apply theme to all active session widgets
        for i in range(self.session_list.count()):
            w = self.session_list.itemWidget(self.session_list.item(i))
            if isinstance(w, SessionItemWidget):
                w.apply_widget_theme()
                
        # apply to examples tab
        self.examples_tab.apply_theme()


    # ── load example from gallery ──
    def load_from_example(self, graph_dict: dict):
        """Deserialize and display an example graph on the canvas, then switch view."""
        self.node_scene.deserialize(graph_dict)
        self.rank_list.clear()
        self.auto_sort_topology()
        self.log("Example loaded into canvas.", "#A6E3A1")
        self.btn_nav_builder.setChecked(True)
        self.main_stack.setCurrentIndex(0) # Switch to Builder view page

    # ── chat history / sessions ──
    def refresh_session_list(self):
        self.session_list.clear()
        active_id = self.session_manager.active_session_id
        # Sort by timestamp descending
        sorted_sessions = sorted(self.session_manager.sessions.values(), key=lambda s: s.timestamp, reverse=True)
        for sess in sorted_sessions:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, sess.id)
            item.setSizeHint(QSize(200, 40)) 
            self.session_list.addItem(item)
            
            widget = SessionItemWidget(sess.name, sess.id, sess.alias, sess.buffer)
            widget.deleteRequested.connect(self.delete_session_by_id)
            widget.sizeChanged.connect(lambda h, i=item: i.setSizeHint(QSize(200, h)))
            self.session_list.setItemWidget(item, widget)

            if sess.id == active_id:
                item.setSelected(True)
                # Sync buffer_store with active session
                GraphExecutor.buffer_store = sess.buffer

    def on_session_clicked(self, item):
        session_id = item.data(Qt.ItemDataRole.UserRole)
        self.session_manager.set_active_session(session_id)
        sess = self.session_manager.get_active_session()
        if sess:
            GraphExecutor.buffer_store = sess.buffer
            self.log(f"Switched to session: {sess.name}", "#CBA6F7")
            # Update output if history node exists? 
            # For now just log it.
        self.refresh_session_list()

    def create_new_session(self):
        new_sess = self.session_manager.create_session()
        self.log(f"Created new session: {new_sess.name}", "#A6E3A1")
        self.refresh_session_list()

    def rename_session_item(self):
        item = self.session_list.currentItem()
        if not item: return
        session_id = item.data(Qt.ItemDataRole.UserRole)
        sess = self.session_manager.sessions.get(session_id)
        if not sess: return
        from PyQt6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(self, "Rename Session", "Enter new name:", text=sess.name)
        if ok and new_name.strip():
            self.session_manager.rename_session(session_id, new_name.strip())
            self.refresh_session_list()

    def delete_session_by_id(self, session_id):
        self.session_manager.delete_session(session_id)
        self.log(f"Session deleted.", "#F38BA8")
        self.refresh_session_list()

    def delete_session_item(self):
        """Older method, still exists for safety but not used in UI buttons."""
        item = self.session_list.currentItem()
        if not item: return
        self.delete_session_by_id(item.data(Qt.ItemDataRole.UserRole))

    def set_session_alias_item(self):
        item = self.session_list.currentItem()
        if not item: return
        session_id = item.data(Qt.ItemDataRole.UserRole)
        sess = self.session_manager.sessions.get(session_id)
        if not sess: return
        from PyQt6.QtWidgets import QInputDialog
        new_alias, ok = QInputDialog.getText(self, "Set Session Alias", "Enter alias (display name):", text=sess.alias)
        if ok:
            self.session_manager.set_session_alias(session_id, new_alias.strip())
            self.refresh_session_list()

    def show_session_context_menu(self, pos):
        item = self.session_list.itemAt(pos)
        if not item: return
        from PyQt6.QtWidgets import QMenu
        menu = QMenu()
        act_alias = menu.addAction("\U0001f3f7 Set Alias")
        act_del = menu.addAction("\U0001f5d1 Delete")
        
        action = menu.exec(self.session_list.mapToGlobal(pos))
        if action == act_alias: self.set_session_alias_item()
        elif action == act_del: self.delete_session_by_id(item.data(Qt.ItemDataRole.UserRole))

    def toggle_history_sidebar(self):
        visible = self.history_sidebar.isVisible()
        self.history_sidebar.setVisible(not visible)

    # ── architecture diagram ──
    def show_arch_diagram(self):
        text = generate_arch_diagram(self.node_scene)
        dlg = ArchDialog(text, self); dlg.exec()

    # ── default pipeline ──
    def build_default_architecture(self):
        self.node_scene.clear(); self.node_scene.undo_stack.clear(); self.rank_list.clear()
        self.log("Building default RAG template...", "#89B4FA")
        n1 = self.node_scene._raw_add_node("PDF Loader", QPointF(-550, 50))
        n2 = self.node_scene._raw_add_node("FAISS DB", QPointF(-200, 50))
        n3 = self.node_scene._raw_add_node("Query Input", QPointF(150, -100))
        n4 = self.node_scene._raw_add_node("Ollama LLM", QPointF(500, -100))
        n5 = self.node_scene._raw_add_node("Response Output", QPointF(500, 150))
        def link(a, b, op=0, ip=0):
            s1, s2 = a.outputs[op], b.inputs[ip]
            e = EdgeItem(s1, s2); s1.edges.append(e); s2.edges.append(e)
            self.node_scene.addItem(e); e.update_path()
        link(n1, n2); link(n2, n4); link(n3, n4); link(n4, n5)
        self.auto_sort_topology()

    # ── node actions ──
    def handle_node_action(self, nid, action, params):
        node = self.node_scene._find_node(nid)
        if not node: return
        if action == "upload_pdf":
            files, _ = QFileDialog.getOpenFileNames(self, "Select PDFs", "", "PDF (*.pdf)")
            if files:
                self.pdf_list = files
                node.set_data("file", f"{len(files)} file(s)")
                self.log(f"PDF Loader: {len(files)} documents loaded.", "#F9E2AF")
        elif action == "build_db":
            if not self.pdf_list:
                self.log("ERROR: No PDFs loaded.", "#F38BA8"); return
            node.set_data("status", "\U0001f534 Indexing...")
            self.log("FAISS: Indexing...", "#89B4FA")
            self.ingest_th = IngestThread(self.pdf_list)
            self.ingest_th.finished.connect(lambda s: self._on_db_built(node, s))
            self.ingest_th.error.connect(lambda e: self._on_db_err(node, e))
            self.ingest_th.start()
        elif action == "run_query_node":
            # Quick Run — preserves cache+buffer, traces the graph
            q = params.get("query", "").strip()
            if not q: self.log("WARNING: Empty query.", "#F9E2AF"); return
            self.log(f"[Quick Run] '{q}'", "#CBA6F7")
            self._launch_graph_run(q, reset_buffer=False)
        elif action == "clear_db":
            if self.engine.clear_db():
                node.set_data("status", "\U0001f534 Disconnected")
                self.log("FAISS DB: Index cleared and deleted from disk.", "#F38BA8")
            else:
                self.log("FAISS DB: Nothing to clear.", "#A6ADC8")
        elif action == "clear_buffer":
            from graph_executor import GraphExecutor
            GraphExecutor.buffer_store = []
            self.log("Buffer: Conversation history cleared.", "#F38BA8")
            node.set_data("status", "\U0001f7e2 Cleared")
        elif action == "clear_cache":
            from graph_executor import GraphExecutor
            GraphExecutor.cache_store = {}
            self.log("Cache: Semantic cache cleared.", "#F38BA8")
            node.set_data("status", "\U0001f7e2 Cleared")
        elif action == "inject_seed":
            seed = params.get("seed", "").strip()
            if seed:
                from graph_executor import GraphExecutor
                for line in seed.split("\n"):
                    line = line.strip()
                    if line and line not in GraphExecutor.buffer_store:
                        GraphExecutor.buffer_store.insert(0, line)
                self.log(f"Seed Buffer Loader: injected {len(seed.splitlines())} turns into buffer.", "#F9E2AF")
                node.set_data("status", f"\U0001f7e2 {len(seed.splitlines())} turns seeded")
            else:
                self.log("Seed Buffer Loader: no seed text provided.", "#F9E2AF")

    def _collect_node_configs(self):
        """Read runtime config values from each node's embedded widgets."""
        configs = {}
        for item in self.node_scene.items():
            if not isinstance(item, NodeItem): continue
            cfg = {}
            if item.node_type == "FAISS DB" and hasattr(item, 'spin_k'):
                cfg["k"] = item.spin_k.value()
            elif item.node_type == "Buffer":
                if hasattr(item, 'combo_mode'): cfg["mode"] = item.combo_mode.currentText()
                if hasattr(item, 'spin_win'): cfg["window_size"] = item.spin_win.value()
            elif item.node_type == "Cache":
                if hasattr(item, 'spin_ttl'): cfg["ttl_seconds"] = item.spin_ttl.value()
                if hasattr(item, 'spin_max'): cfg["max_entries"] = item.spin_max.value()
            elif item.node_type == "Router":
                if hasattr(item, 'combo_rt'): cfg["routing_type"] = item.combo_rt.currentText()
                if hasattr(item, 'line_kw'): cfg["keyword"] = item.line_kw.text()
            # ── New node configs ──────────────────────────────────────────
            elif item.node_type == "Prompt Template":
                if hasattr(item, 'tmpl_edit'): cfg["template"] = item.tmpl_edit.toPlainText()
            elif item.node_type == "System Message":
                if hasattr(item, 'sys_edit'): cfg["message"] = item.sys_edit.toPlainText()
            elif item.node_type == "Reranker":
                if hasattr(item, 'spin_topk'): cfg["top_k"] = item.spin_topk.value()
            elif item.node_type == "Memory Formatter":
                if hasattr(item, 'spin_maxtok'): cfg["max_tokens"] = item.spin_maxtok.value()
                if hasattr(item, 'cb_summarize'): cfg["summarize"] = item.cb_summarize.isChecked()
            elif item.node_type == "Conversation Starter":
                if hasattr(item, 'starter_edit'): cfg["text"] = item.starter_edit.toPlainText()
            elif item.node_type == "Score Filter":
                if hasattr(item, 'spin_thresh'): cfg["threshold"] = item.spin_thresh.value()
            elif item.node_type == "Top-K Retriever":
                if hasattr(item, 'spin_topk_ret'): cfg["k"] = item.spin_topk_ret.value()
            elif item.node_type == "Seed Buffer Loader":
                if hasattr(item, 'seed_edit'): cfg["seed"] = item.seed_edit.toPlainText()
            if cfg:
                configs[item.node_id] = cfg
        return configs

    def _launch_graph_run(self, query_text, reset_buffer=False):
        """Launch graph execution in a background thread."""
        graph_data = self.node_scene.serialize()
        configs = self._collect_node_configs()

        # Set all nodes to waiting
        for item in self.node_scene.items():
            if isinstance(item, NodeItem):
                item.set_data("status", "\U0001f7e1 Queued")

        self.current_query = query_text
        self.graph_thread = GraphRunThread(graph_data, self.engine, configs, reset_buffer)
        self.graph_thread.query_text = query_text
        self.graph_thread.log_signal.connect(self.log)
        self.graph_thread.status_signal.connect(self._update_node_status)
        self.graph_thread.finished.connect(lambda txt: self._on_graph_done(txt))
        self.graph_thread.error.connect(lambda e: self.log(f"Graph Error: {e}", "#F38BA8"))
        self.graph_thread.start()

    def _update_node_status(self, node_id, status_text):
        node = self.node_scene._find_node(node_id)
        if not node: return
        
        if status_text.startswith("PEEK:"):
            # Detailed data preview for simulation
            node.set_data("peek", status_text.replace("PEEK:", ""))
        elif status_text.startswith("\U0001f50e"): # Inspect icon 🔎
            # Debug inspector also goes to peek
            node.set_data("peek", status_text)
            node.set_data("data", status_text) # Compatibility for Debug node
        else:
            # Normal status dot update
            node.set_data("status", status_text)

    def _on_graph_done(self, result_text):
        # Push result into all Response Output nodes
        for item in self.node_scene.items():
            if isinstance(item, NodeItem) and item.node_type == "Response Output":
                item.set_data("answer", result_text)
                item.set_data("status", "\U0001f7e2 Done")
        self.log(f"Pipeline complete. Output: {len(result_text)} chars.", "#A6E3A1")
        # Save updated buffer to active session
        active_sess = self.session_manager.get_active_session()
        if active_sess:
            # Auto-log the interaction if it's not already logged
            entry_user = f"User: {self.current_query}"
            entry_ai = f"AI: {result_text}"
            if entry_user not in GraphExecutor.buffer_store:
                GraphExecutor.buffer_store.append(entry_user)
            if entry_ai not in GraphExecutor.buffer_store:
                GraphExecutor.buffer_store.append(entry_ai)
                
            self.session_manager.update_session_buffer(active_sess.id, GraphExecutor.buffer_store)
            # Refresh UI to show new lines
            self.refresh_session_list()

    # ── full pipeline ──
    def run_global_pipeline(self):
        # Step 1: Ensure FAISS DB is loaded before anything
        if not self.engine.vector_db:
            self.engine.load_db()
            if not self.engine.vector_db:
                self.log("ERROR: No FAISS index found. Upload PDFs and click 'Index Database' first.", "#F38BA8")
                return
            else:
                self.log("FAISS DB loaded from disk.", "#89B4FA")

        qn = [i for i in self.node_scene.items() if isinstance(i, NodeItem) and i.node_type == "Query Input"]
        if not qn:
            self.log("ERROR: No Query Input node.", "#F38BA8"); return
        qt = qn[0].get_data("query")
        if not qt: self.log("WARNING: Empty query.", "#F9E2AF"); return
        self.log(f"[Full Pipeline] '{qt}' (buffer reset, DB verified)", "#A6E3A1")
        self._launch_graph_run(qt, reset_buffer=True)

    # ── DB callbacks ──
    def _on_db_built(self, node, stats):
        node.set_data("status", "\U0001f7e2 DB Built")
        self.try_load_db()
        self.log(f"FAISS indexed in {stats['indexing_time']}s", "#A6E3A1")

    def _on_db_err(self, node, err):
        node.set_data("status", "\U0001f534 Error"); self.log(f"DB Error: {err}", "#F38BA8")

    def try_load_db(self):
        if self.engine.load_db():
            for i in self.node_scene.items():
                if isinstance(i, NodeItem) and i.node_type == "FAISS DB":
                    i.set_data("status", "\U0001f7e2 Connected")
            self.log("FAISS DB loaded.", "#89B4FA")

    def save_architecture(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save", "", "JSON (*.json)")
        if p:
            with open(p, 'w') as f: json.dump(self.node_scene.serialize(), f, indent=2)
            self.log(f"Saved to {p}", "#89B4FA")

    def load_architecture(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load", "", "JSON (*.json)")
        if p:
            with open(p, 'r') as f: self.node_scene.deserialize(json.load(f))
            self.auto_sort_topology(); self.log(f"Loaded from {p}", "#89B4FA")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = RAGWindow(); w.show()
    sys.exit(app.exec())
