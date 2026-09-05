from finrl.benchmark.agent import ConversationWindow


def _entry(text: str) -> str:
    return f"Model Output:\n{text}"


def test_window_empty_when_no_entries():
    w = ConversationWindow(max_tokens=1000, keep_last_n=1)
    assert w.get_history_str() == ""
    assert w.history_tokens_now == 0
    assert w.entries_now == 0


def test_window_keeps_all_below_budget():
    w = ConversationWindow(max_tokens=10000, keep_last_n=2)
    for i in range(10):
        w.append(_entry(f"step {i}"))
    assert w.entries_now == 10
    assert w.dropped_exchanges == 0
    assert w.truncation_count == 0
    assert "step 0" in w.get_history_str()
    assert "step 9" in w.get_history_str()


def test_window_prunes_oldest_when_over_budget():
    w = ConversationWindow(max_tokens=200, keep_last_n=1)
    for i in range(50):
        w.append(_entry("x" * 80))
    assert w.entries_now < 50
    assert w.dropped_exchanges > 0
    assert w.truncation_count > 0
    history = w.get_history_str()
    assert "truncated to fit 200-token context budget" in history
    assert "step 49" in history or "xxxxxxxx" in history


def test_window_always_keeps_recent_exchanges():
    w = ConversationWindow(max_tokens=45, keep_last_n=2)
    for i in range(6):
        w.append(_entry(f"step{i}-" + "x" * 40))
    history = w.get_history_str()
    assert "step5" in history
    assert "step4" in history
    assert "step0" not in history


def test_window_tracks_full_tokens():
    w = ConversationWindow(max_tokens=1000, keep_last_n=1)
    w.append("a" * 40)
    w.append("b" * 44)
    assert w.full_tokens_total == 10 + 11
    assert w.history_tokens_now == 21


def test_window_rejects_bad_budget():
    try:
        ConversationWindow(max_tokens=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for max_tokens=0")