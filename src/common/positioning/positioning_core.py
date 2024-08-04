class Position:
    def __init__(self, name: str, count: int) -> None:
        self.name = name
        self.count = count

    def hello_instance_name(self) -> None:
        print(self.name)

    def count_add(self) -> None:
        self.count = self.count + 1
        print(self.count)

    @staticmethod
    def hello_world() -> str:
        print("Hello World!")
        return "Hello World!"

    @staticmethod
    def hello_world2(output: str) -> None:
        print(output)
