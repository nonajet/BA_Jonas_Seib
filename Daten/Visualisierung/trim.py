import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [0, 0, 0],
                   [1, 1, 1]])

# Finde Indizes der Zeilen, die nicht nur Nullen enthalten (am Ende)
non_zero_rows = np.any(matrix != 0, axis=1)
end_index = len(matrix) - np.argmax(non_zero_rows[::-1])  # Reverse-Index

# Entferne Nullzeilen am Ende
trimmed_matrix = matrix[:end_index]

# Zeige die resultierende Matrix an
print(trimmed_matrix)
