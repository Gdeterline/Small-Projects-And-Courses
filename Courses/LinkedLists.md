# Linked Lists in Python – Theoretical Course with Python Exercises

---

## 1. Introduction

### 1.1 What is a Linked List?

* A **linked list** is a data structure where elements (nodes) are connected using **pointers (references)**.
* Each **node** contains:

  * `data`: the actual value.
  * `next`: a reference to the next node.

Example diagram of a singly linked list:

```
[Head] -> [Data|Next] -> [Data|Next] -> [Data|Next] -> None
```

In Python, we implement nodes with **classes**, and use `None` to signify the end of the list.
* The first node is called the **head**.
* The last node points to `None`, indicating the end of the list. It is called the **tail**.
* Linked lists can be **singly linked** (one direction) or **doubly linked** (two directions), or even **circular** (last node points back to head).

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

A simple node class with `data` and `next` pointer. The `next` is initialized to `None`.

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
```

### 2.2 LinkedList Class

The linked list class manages the head of the list. It provides methods for common operations like insertion, deletion, and traversal.

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

A doubly linked is a linked list where each node has two pointers: one to the next node and one to the previous node.
This allows traversal in both directions.
It can be particularly useful for certain applications like navigation systems, undo functionality in applications, and more.

### 4.1 Node

The node structure for a doubly linked list includes `data`, `next`, and `prev` pointers.

### 4.2 Append and Prepend

To append, create a new node, set its `prev` to the current tail, and update the tail's `next` to this new node.
To prepend, create a new node, set its `next` to the current head, and update the head's `prev` to this new node.

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

A circular linked list is a linked list where the last node points back to the head, forming a circle.
In a circular linked list, there is no `None` at the end; instead, the last node's `next` points to the head node.

### 5.2 Traversal

To traverse a circular linked list, start from the head and continue until you reach the head again. Use a `do-while` style loop to ensure the head is processed.

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
