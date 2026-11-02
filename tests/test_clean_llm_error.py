"""Provider Test should show the API's message, not the client object's dump."""


class _ApiError(Exception):
    def __init__(self, message, body):
        super().__init__(message)
        self.body = body


def test_clean_llm_error_prefers_the_api_message():
    from core.llm_error import clean_llm_error as _clean_llm_error
    err = _ApiError(
        "Error code: 400 - {'error': {'message': 'huge client dump'}}",
        {"error": {"message": 'Unknown name "google": Cannot find field.', "code": 400}},
    )
    assert _clean_llm_error(err) == 'Unknown name "google": Cannot find field.'


def test_clean_llm_error_falls_back_to_the_exception_text():
    from core.llm_error import clean_llm_error as _clean_llm_error
    assert _clean_llm_error(RuntimeError("connection refused")) == "connection refused"
