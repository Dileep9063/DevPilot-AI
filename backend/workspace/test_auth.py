from auth import login


def test_login():
    result = login("admin", "1234")

    assert result is True