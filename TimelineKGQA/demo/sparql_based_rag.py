import mesop as me

from TimelineKGQA.rag.gpt import RAGRank


def on_load(e: me.LoadEvent):
    me.set_theme_mode("system")


@me.stateclass
class State:
    question_id: int = None
    in_progress: bool
    rewrite: str
    rewrite_message_index: int
    preview_rewrite: str
    preview_original: str
    modal_open: bool


def on_blur(e: me.InputBlurEvent):
    s = me.state(State)
    try:
        s.question_id = int(e.value)
    except ValueError:
        s.question_id = None


@me.page(
    security_policy=me.SecurityPolicy(
        allowed_iframe_parents=["https://google.github.io", "https://huggingface.co"]
    ),
    path="/semantic_parsing",
    title="Semantic Parsing QA Exploration",
    on_load=on_load,
)
def page():
    rag = RAGRank(
        table_name="unified_kg_icews_actor",
        host="localhost",
        port=5433,
        user="tkgqa",
        password="tkgqa",
        db_name="tkgqa",
    )
    state = me.state(State)

    with me.box(style=_STYLE_CHAT_INPUT_BOX):
        with me.box(style=me.Style(flex_grow=1)):
            me.input(
                label=_LABEL_INPUT,
                # Workaround: update key to clear input.
                on_input=on_chat_input,
                on_enter=on_click_submit_chat_msg,
                style=_STYLE_CHAT_INPUT,
            )
        with me.content_button(
            color="primary",
            type="flat",
            disabled=state.in_progress,
            on_click=on_click_submit_chat_msg,
            style=_STYLE_CHAT_BUTTON,
        ):
            me.icon(_LABEL_BUTTON_IN_PROGRESS if state.in_progress else _LABEL_BUTTON)


def on_chat_input(e: me.InputEvent):
    """Capture chat text input."""
    state = me.state(State)
    state.question_id = e.value


def on_rewrite_input(e: me.InputEvent):
    """Capture rewrite text input."""
    state = me.state(State)
    state.preview_rewrite = e.value


def on_click_submit_chat_msg(e: me.ClickEvent | me.InputEnterEvent):
    """Handles submitting a chat message."""
    state = me.state(State)
    if state.in_progress or not state.input:
        return
    input = state.input
    state.input = ""
    yield


_STYLE_CHAT_INPUT_BOX = me.Style(
    padding=me.Padding(top=30), display="flex", flex_direction="row"
)

_LABEL_INPUT = "Enter your question id"
# Constants

_TITLE = "LLM Rewriter"

_ROLE_USER = "user"
_ROLE_ASSISTANT = "assistant"

_BOT_USER_DEFAULT = "mesop-bot"

# Styles

_COLOR_BACKGROUND = "#f0f4f8"
_COLOR_CHAT_BUBBLE_YOU = "#f2f2f2"
_COLOR_CHAT_BUBBLE_BOT = "#ebf3ff"
_COLOR_CHAT_BUUBBLE_EDITED = "#f2ebff"

_DEFAULT_PADDING = me.Padding.all(20)
_DEFAULT_BORDER_SIDE = me.BorderSide(width="1px", style="solid", color="#ececec")

_LABEL_BUTTON = "send"
_LABEL_BUTTON_IN_PROGRESS = "pending"
_LABEL_INPUT = "Enter your prompt"

_STYLE_INPUT_WIDTH = me.Style(width="100%")

_STYLE_APP_CONTAINER = me.Style(
    background=_COLOR_BACKGROUND,
    display="grid",
    height="100vh",
    grid_template_columns="repeat(1, 1fr)",
)
_STYLE_TITLE = me.Style(padding=me.Padding(left=10))
_STYLE_CHAT_BOX = me.Style(
    height="100%",
    overflow_y="scroll",
    padding=_DEFAULT_PADDING,
    margin=me.Margin(bottom=20),
    border_radius="10px",
    border=me.Border(
        left=_DEFAULT_BORDER_SIDE,
        right=_DEFAULT_BORDER_SIDE,
        top=_DEFAULT_BORDER_SIDE,
        bottom=_DEFAULT_BORDER_SIDE,
    ),
)
_STYLE_CHAT_INPUT = me.Style(width="100%")
_STYLE_CHAT_INPUT_BOX = me.Style(
    padding=me.Padding(top=30), display="flex", flex_direction="row"
)
_STYLE_CHAT_BUTTON = me.Style(margin=me.Margin(top=8, left=8))
_STYLE_CHAT_BUBBLE_NAME = me.Style(
    font_weight="bold",
    font_size="12px",
    padding=me.Padding(left=15, right=15, bottom=5),
)
_STYLE_CHAT_BUBBLE_PLAINTEXT = me.Style(margin=me.Margin.symmetric(vertical=15))

_STYLE_MODAL_CONTAINER = me.Style(
    background="#fff",
    margin=me.Margin.symmetric(vertical="0", horizontal="auto"),
    width="min(1024px, 100%)",
    box_sizing="content-box",
    height="100vh",
    overflow_y="scroll",
    box_shadow=("0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f"),
)

_STYLE_MODAL_CONTENT = me.Style(margin=me.Margin.all(20))

_STYLE_PREVIEW_CONTAINER = me.Style(
    display="grid",
    grid_template_columns="repeat(2, 1fr)",
)

_STYLE_PREVIEW_ORIGINAL = me.Style(color="#777", padding=_DEFAULT_PADDING)

_STYLE_PREVIEW_REWRITE = me.Style(
    background=_COLOR_CHAT_BUUBBLE_EDITED, padding=_DEFAULT_PADDING
)
