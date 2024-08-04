from positioning import Position


def main() -> None:
    # Position.hello_world()
    # obj = Position()
    # obj.hello_world2("hello")
    obj = Position("hogehoge", 2)
    obj.hello_instance_name()
    obj.count_add()
    obj.count_add()
    obj.count_add()


if __name__ == "__main__":
    main()
