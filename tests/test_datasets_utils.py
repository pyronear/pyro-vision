import pytest

from pyrovision.datasets import utils


@pytest.mark.parametrize(
    "url, max_base_length, expected_name",
    [
        ["https://pyronear.org/img/logo_letters.png", None, "logo_letters.png"],
        ["https://pyronear.org/img/logo_letters.png?height=300", None, "logo_letters.png"],
        ["https://pyronear.org/img/logo_letters.png?height=300&width=400", None, "logo_letters.png"],
        ["https://pyronear.org/img/logo_letters", None, "logo_letters.jpg"],
        ["https://pyronear.org/img/very_long_file_name.png", 10, "very_long_.png"],
    ],
)
def test_get_fname(url, max_base_length, expected_name):
    assert utils.get_fname(url, max_base_length=max_base_length) == expected_name


@pytest.mark.parametrize(
    "arr, num_threads, progress, expected",
    [
        [[1, 2, 3, 4, 5], None, False, [1, 4, 9, 16, 25]],
        [[1, 2, 3, 4, 5], None, True, [1, 4, 9, 16, 25]],
        [[1, 2, 3, 4, 5], 1, False, [1, 4, 9, 16, 25]],
        [[1, 2, 3, 4, 5], 2, False, [1, 4, 9, 16, 25]],
    ],
)
def test_parallel(arr, num_threads, progress, expected):
    assert list(utils.parallel(lambda x: x**2, arr, num_threads=num_threads, progress=progress)) == expected
