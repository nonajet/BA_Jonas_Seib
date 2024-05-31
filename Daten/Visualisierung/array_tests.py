if __name__ == '__main__':
    list1 = [1, 2, 3, 4]
    list2 = ['a', 'b', 'c', 'd']

    for item1, item2 in zip(list1, list2):
        print(f"Item from list1: {item1}, Item from list2: {item2}")
