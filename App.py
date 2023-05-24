import tkinter as tk
from ttkthemes import ThemedTk
import metodos as methods

# Style variables
font = "Arial"
size = 16
main_bg = "#%02x%02x%02x" % (255, 255, 151)
button_bg = "#%02x%02x%02x" % (int(255 * 0.9), int(255 * 0.9), int(151 * 0.9))
sidebar_bg = "#%02x%02x%02x" % (255, 179, 71)
sidebutton_bg = "#%02x%02x%02x" % (int(255 * 0.9), int(179 * 0.9), int(71 * 0.9))


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
window = ThemedTk(theme="radiant")
window.title("Métodos Análisis Numérico")
window.configure(background=main_bg)
window.state("zoomed")
window.geometry("800x500")

# ---- Main Screen ----
main_screen = tk.Frame(window, bg=main_bg)
current_screen = main_screen

title_label = tk.Label(
    main_screen,
    text="Métodos de Análisis Numérico",
    font=(font, 20, "bold"),
    bg=main_bg,
)
title_label.pack(pady=20)

incremental_search_button = tk.Button(
    main_screen,
    text="Método de Búsqueda Incremental",
    font=(font, 14),
    bg=button_bg,
    command=lambda: switch_screen(incremental_search),
)
incremental_search_button.pack()

bisection_button = tk.Button(
    main_screen,
    text="Método de Bisección",
    font=(font, 14),
    bg=button_bg,
    command=lambda: switch_screen(bisection),
)
bisection_button.pack()

regla_falsa_button = tk.Button(
    main_screen,
    text="Método de la Regla Falsa",
    font=(font, 14),
    bg=button_bg,
    command=lambda: switch_screen(regla_falsa),
)
regla_falsa_button.pack()

# ---- Sidebar ----
sidebar = tk.Frame(window, bg=sidebar_bg, width=150)

sidebar_label = tk.Label(
    sidebar,
    text="Menú",
    font=(font, size),
    bg=sidebar_bg,
    pady=10,
)
sidebar_label.pack()

return_button = tk.Button(
    sidebar,
    text="Volver a Principal",
    font=(font, 12),
    bg=sidebutton_bg,
    command=return_to_main,
)
return_button.pack(pady=10)

sidebar.pack(side="left", fill="y")


# ---- Result Screen ----
result_screen = tk.Frame(window, bg=main_bg)

result_label = tk.Label(
    result_screen,
    text="Resultado:",
    font=(font, size),
    bg=main_bg,
)
result_label.pack(pady=20)

# ---- Matrix Screen ----
matrix_screen = tk.Frame(window, bg=main_bg)

name_label_output = tk.Label(
    matrix_screen,
    text="Tamaño de la matriz:",
    font=(font, size),
    bg=main_bg,
)
name_label_output.pack(pady=20)
name_entry = tk.Entry(matrix_screen, font=(font, 14))
name_entry.pack()

next_button = tk.Button(
    matrix_screen,
    text="Continuar",
    font=(font, 14),
    bg=button_bg,
    command=save_matrix,
)
next_button.pack(pady=20)

# ---- Incremental Search Screen ----
incremental_search = tk.Frame(window, bg=main_bg)

is1_title_label = tk.Label(
    incremental_search,
    text="Método de Búsqueda Incremental",
    font=(font, 20, "bold"),
    bg=main_bg,
)
is1_title_label.pack(pady=20)

is1_f = tk.Label(
    incremental_search,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
is1_f.pack(pady=10)
is1_f_entry = tk.Entry(incremental_search, font=(font, 14))
is1_f_entry.pack()

is1_xinit = tk.Label(
    incremental_search,
    text="X inicial:",
    font=(font, size),
    bg=main_bg,
)
is1_xinit.pack(pady=10)
is1_xinit_entry = tk.Entry(incremental_search, font=(font, 14))
is1_xinit_entry.pack()

is1_dx = tk.Label(
    incremental_search,
    text="Derivada:",
    font=(font, size),
    bg=main_bg,
)
is1_dx.pack(pady=10)
is1_dx_entry = tk.Entry(incremental_search, font=(font, 14))
is1_dx_entry.pack()

is1_n = tk.Label(
    incremental_search,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
is1_n.pack(pady=10)
is1_n_entry = tk.Entry(incremental_search, font=(font, 14))
is1_n_entry.pack()

is1_show_button = tk.Button(
    incremental_search,
    text="Resolver",
    font=(font, 14),
    bg=button_bg,
    command=lambda: show_result(
        methods.incremental_search(
            is1_f_entry.get(),
            is1_xinit_entry.get(),
            is1_dx_entry.get(),
            is1_n_entry.get(),
        )
    ),
)
is1_show_button.pack(pady=20)

# ---- Bisection Screen ----
bisection = tk.Frame(window, bg=main_bg)

b2_title_label = tk.Label(
    bisection,
    text="Método de Bisección",
    font=(font, 20, "bold"),
    bg=main_bg,
)
b2_title_label.pack(pady=20)

b2_f = tk.Label(
    bisection,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
b2_f.pack(pady=10)
b2_f_entry = tk.Entry(bisection, font=(font, 14))
b2_f_entry.pack()

b2_xi = tk.Label(
    bisection,
    text="X Inferior:",
    font=(font, size),
    bg=main_bg,
)
b2_xi.pack(pady=10)
b2_xi_entry = tk.Entry(bisection, font=(font, 14))
b2_xi_entry.pack()

b2_xs = tk.Label(
    bisection,
    text="X Superior:",
    font=(font, size),
    bg=main_bg,
)
b2_xs.pack(pady=10)
b2_xs_entry = tk.Entry(bisection, font=(font, 14))
b2_xs_entry.pack()

b2_tol = tk.Label(
    bisection,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
b2_tol.pack(pady=10)
b2_tol_entry = tk.Entry(bisection, font=(font, 14))
b2_tol_entry.pack()

b2_n = tk.Label(
    bisection,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
b2_n.pack(pady=10)
b2_n_entry = tk.Entry(bisection, font=(font, 14))
b2_n_entry.pack()

b2_show_button = tk.Button(
    bisection,
    text="Resolver",
    font=(font, 14),
    bg=button_bg,
    command=lambda: show_result(
        methods.biseccion(
            b2_f_entry.get(),
            b2_xi_entry.get(),
            b2_xs_entry.get(),
            b2_tol_entry.get(),
            b2_n_entry.get(),
        )
    ),
)
b2_show_button.pack(pady=20)

# ---- regla_falsa Screen ----
regla_falsa = tk.Frame(window, bg=main_bg)

rf3_title_label = tk.Label(
    regla_falsa,
    text="Método de la Regla Falsa",
    font=(font, 20, "bold"),
    bg=main_bg,
)
rf3_title_label.pack(pady=20)

rf3_f = tk.Label(
    regla_falsa,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
rf3_f.pack(pady=10)
rf3_f_entry = tk.Entry(regla_falsa, font=(font, 14))
rf3_f_entry.pack()

rf3_terror = tk.Label(
    regla_falsa,
    text="t error:",
    font=(font, size),
    bg=main_bg,
)
rf3_terror.pack(pady=10)
rf3_terror_entry = tk.Entry(regla_falsa, font=(font, 14))
rf3_terror_entry.pack()

rf3_xi = tk.Label(
    regla_falsa,
    text="X Inferior:",
    font=(font, size),
    bg=main_bg,
)
rf3_xi.pack(pady=10)
rf3_xi_entry = tk.Entry(regla_falsa, font=(font, 14))
rf3_xi_entry.pack()

rf3_xs = tk.Label(
    regla_falsa,
    text="X Superior:",
    font=(font, size),
    bg=main_bg,
)
rf3_xs.pack(pady=10)
rf3_xs_entry = tk.Entry(regla_falsa, font=(font, 14))
rf3_xs_entry.pack()

rf3_tol = tk.Label(
    regla_falsa,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
rf3_tol.pack(pady=10)
rf3_tol_entry = tk.Entry(regla_falsa, font=(font, 14))
rf3_tol_entry.pack()

rf3_n = tk.Label(
    regla_falsa,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
rf3_n.pack(pady=10)
rf3_n_entry = tk.Entry(regla_falsa, font=(font, 14))
rf3_n_entry.pack()

rf3_show_button = tk.Button(
    regla_falsa,
    text="Resolver",
    font=(font, 14),
    bg=button_bg,
    command=lambda: show_result(
        methods.biseccion(
            rf3_f_entry.get(),
            rf3_terror_entry.get(),
            rf3_xi_entry.get(),
            rf3_xs_entry.get(),
            rf3_tol_entry.get(),
            rf3_n_entry.get(),
        )
    ),
)
rf3_show_button.pack(pady=20)


# Initially show the input screen
main_screen.pack()

# Run the application
window.mainloop()
