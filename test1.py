from matrix import load_data
from matrix import build_matrix

text, numbers = load_data("amazon_cells_labelled.txt")
print("Total sentences:", len(text))
print("First sentence:", text[0])
D, words = build_matrix(text)
print("Matrix dimension:", D.shape)
print("Vocab size:", len(words))