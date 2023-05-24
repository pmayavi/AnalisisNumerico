import tkinter as tk
from ttkthemes import ThemedTk
import metodos as methods


def switch_screen(screen):
    # Hide the current screen
    global current_screen
    current_screen.pack_forget()

    # Show the selected screen
    current_screen = screen
    current_screen.pack()


def get_matrix(screen):
    # Hide the current screen
    main_screen.pack_forget()

    # Show the selected screen
    global current_screen
    current_screen = screen
    current_screen.pack()


def save_matrix():
    # Hide the current screen
    matrix_screen.pack_forget()

    # Show the selected screen
    global current_screen
    current_screen.pack()


def show_result(res):
    result_label.config(text=str(res))
    result_screen.pack()


def return_to_main():
    global current_screen
    # Hide the current screen
    current_screen.pack_forget()
    current_screen = main_screen

    # Show the main screen
    main_screen.pack()


# Create a new themed window
window = ThemedTk(theme="clam")
window.title("Métodos Análisis Numérico")
window.geometry("800x500")

# Main Screen
main_screen = tk.Frame(window, bg="#f0f0f0")
current_screen = main_screen

title_label = tk.Label(
    main_screen,
    text="Métodos de Análisis Numérico",
    font=("Arial", 20, "bold"),
    bg="#f0f0f0",
)
title_label.pack(pady=20)

submit_button = tk.Button(
    main_screen,
    text="Método de Búsqueda Incremental",
    font=("Arial", 14),
    command=lambda: switch_screen(incremental_search),
)
submit_button.pack()

# Result Screen
result_screen = tk.Frame(window, bg="#f0f0f0")

result_label = tk.Label(
    result_screen,
    text="Resultado:",
    font=("Arial", 16),
    bg="#f0f0f0",
)
result_label.pack(pady=20)

# Matrix Screen
matrix_screen = tk.Frame(window, bg="#f0f0f0")

name_label_output = tk.Label(
    matrix_screen,
    text="Tamaño de la matriz:",
    font=("Arial", 16),
    bg="#f0f0f0",
)
name_label_output.pack(pady=20)
name_entry = tk.Entry(matrix_screen, font=("Arial", 14))
name_entry.pack()

next_button = tk.Button(
    matrix_screen,
    text="Continuar",
    font=("Arial", 14),
    command=save_matrix,
)
next_button.pack(pady=20)

# Incremental Search Screen
incremental_search = tk.Frame(window, bg="#f0f0f0")

is1_f = tk.Label(
    incremental_search,
    text="Función:",
    font=("Arial", 16),
    bg="#f0f0f0",
)
is1_f.pack(pady=10)
is1_f_entry = tk.Entry(incremental_search, font=("Arial", 14))
is1_f_entry.pack()

is1_xinit = tk.Label(
    incremental_search,
    text="X inicial:",
    font=("Arial", 16),
    bg="#f0f0f0",
)
is1_xinit.pack(pady=10)
is1_xinit_entry = tk.Entry(incremental_search, font=("Arial", 14))
is1_xinit_entry.pack()

is1_dx = tk.Label(
    incremental_search,
    text="Derivada:",
    font=("Arial", 16),
    bg="#f0f0f0",
)
is1_dx.pack(pady=10)
is1_dx_entry = tk.Entry(incremental_search, font=("Arial", 14))
is1_dx_entry.pack()

is1_n = tk.Label(
    incremental_search,
    text="Iteraciones:",
    font=("Arial", 16),
    bg="#f0f0f0",
)
is1_n.pack(pady=10)
is1_n_entry = tk.Entry(incremental_search, font=("Arial", 14))
is1_n_entry.pack()

show_button = tk.Button(
    incremental_search,
    text="Resolver",
    font=("Arial", 14),
    command=lambda: show_result(
        methods.incremental_search(
            is1_f_entry.get(),
            is1_xinit_entry.get(),
            is1_dx_entry.get(),
            is1_n_entry.get(),
        )
    ),
)
show_button.pack(pady=20)

# Sidebar
sidebar = tk.Frame(window, bg="#f0f0f0", width=150)

sidebar_label = tk.Label(
    sidebar,
    text="Menú",
    font=("Arial", 16, "bold"),
    bg="#f0f0f0",
    pady=10,
)
sidebar_label.pack()

return_button = tk.Button(
    sidebar,
    text="Volver a Principal",
    font=("Arial", 12),
    command=return_to_main,
)
return_button.pack(pady=10)

sidebar.pack(side="left", fill="y")

# Initially show the input screen
main_screen.pack()

# Run the application
window.mainloop()
