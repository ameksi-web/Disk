"""Small Python example for Delta 1 Flash code training."""


def hello(name: str) -> str:
    return f"Привет, {name}!"


def add(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    print(hello("мир"))
    print("2 + 3 =", add(2, 3))
