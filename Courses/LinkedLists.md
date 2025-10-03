# Linked Lists in Python – Theoretical Course with Python Exercises

---

## 1. Introduction

### 1.1 What is a Linked List?

* A **linked list** is a data structure where elements (nodes) are connected using **pointers (references)**.
* Each **node** contains:

  * `data`: the actual value.
  * `next`: a reference to the next node.

In Python, we implement nodes with **classes**.

---

### 1.2 Comparison with Python Lists

| Feature             | Python List (array-like) | Linked List             |
| ------------------- | ------------------------ | ----------------------- |
| Access time         | O(1) by index            | O(n), must traverse     |
| Insertion in middle | O(n) (shift needed)      | O(1) (if pointer known) |
| Memory layout       | Contiguous               | Scattered in memory     |
| Extra memory        | None                     | Pointer per node        |

---

## 2. Building Blocks in Python

### 2.1 Node Class

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

### 2.2 LinkedList Class

```python
class LinkedList:
    def __init__(self):
        self.head = None
```

---

## 3. Singly Linked List

### 3.1 Traversal

To go through all elements, we start from the head and follow `next` pointers. This is called **traversal**.

---

### 3.2 Insert Operations

* **At the beginning**

To insert at the start, create a new node and point its `next` to the current head. Update head to this new node.


* **At the end**

To insert at the end, traverse to the last node and set its `next` to the new node. To handle an empty list, check if head is `None`.
To identify the last node, check if `next` is `None`.

---

### 3.3 Delete by Value

To delete a node by value, traverse the list while keeping track of the previous node. If the node with the target value is found, update the previous node's `next` to skip the current node.

### 3.4 Delete by Position

To delete a node by position (0-based index), traverse the list to the node just before the target position. Update its `next` to skip the target node.

---

### Exercises (Singly Linked List in Python)

1. Implement a method `length()` that returns the number of nodes.
2. Implement a method `search(key)` that returns `True` if a value exists.
3. Implement a method `get_nth(n)` that returns the nth element’s data (0-based index).
4. Write a method `reverse()` that reverses the linked list.
5. Write a program that:

   * Creates a linked list.
   * Inserts elements `[5, 10, 15, 20]` at the end.
   * Prints the list.
   * Deletes `10`.
   * Prints the list again.

---

## 4. Doubly Linked List

### 4.1 Node

```python
class DNode:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None
```

### 4.2 Append and Prepend

```python
class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = DNode(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        new_node.prev = current

    def prepend(self, data):
        new_node = DNode(data)
        new_node.next = self.head
        if self.head:
            self.head.prev = new_node
        self.head = new_node
```

---

### Exercises (Doubly Linked List in Python)

6. Implement `print_forward()` and `print_backward()` methods.
7. Write a method `delete_by_value(value)` for the doubly linked list.
8. Write a program that:

   * Creates a doubly linked list.
   * Appends `1, 2, 3, 4`.
   * Prepends `0`.
   * Prints forward and backward.

---

## 5. Circular Linked List

### 5.1 Node + Circular List

```python
class CNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class CircularLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = CNode(data)
        if not self.head:
            self.head = new_node
            new_node.next = new_node
            return
        current = self.head
        while current.next != self.head:
            current = current.next
        current.next = new_node
        new_node.next = self.head
```

---

### Exercises (Circular Linked List in Python)

9. Implement `print_list()` for a circular linked list.
10. Implement `delete_by_value(value)` in a circular linked list.
11. Write a program that creates a circular linked list of `[1,2,3,4]` and prints it in a loop-like fashion.

---

## 6. Advanced Exercises (Python Only)

12. Write a function to **merge two sorted singly linked lists** into one sorted list.
13. Write a method `find_middle()` that returns the middle node’s value in a singly linked list.
14. Write a method `has_cycle()` to detect if a linked list has a loop (Floyd’s cycle detection).
15. Convert a Python list `[1,2,3,4,5]` into a linked list using your classes.
