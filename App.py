import tkinter as tk
from ttkthemes import ThemedTk
import metodos as methods

# Style variables
font = "Arial"
size = 16
size1 = 14
size2 = 20
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


def show_result(res):
    matrix_screen1.pack_forget()
    matrix_screen2.pack_forget()
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
    font=(font, size2, "bold"),
    bg=main_bg,
)
title_label.pack(pady=20)

incremental_search_button = tk.Button(
    main_screen,
    text="Método de Búsqueda Incremental",
    font=(font, size1),
    bg=button_bg,
    command=lambda: switch_screen(incremental_search),
)
incremental_search_button.pack()

bisection_button = tk.Button(
    main_screen,
    text="Método de Bisección",
    font=(font, size1),
    bg=button_bg,
    command=lambda: switch_screen(bisection),
)
bisection_button.pack()

regla_falsa_button = tk.Button(
    main_screen,
    text="Método de la Regla Falsa",
    font=(font, size1),
    bg=button_bg,
    command=lambda: switch_screen(regla_falsa),
)
regla_falsa_button.pack()

punto_fijo_button = tk.Button(
    main_screen,
    text="Método del Punto fijo",
    font=(font, size1),
    bg=button_bg,
    command=lambda: switch_screen(punto_fijo),
)
punto_fijo_button.pack()

newton_raphson_button = tk.Button(
    main_screen,
    text="Método de Newton-Raphson",
    font=(font, size1),
    bg=button_bg,
    command=lambda: switch_screen(newton_raphson),
)
newton_raphson_button.pack()

secante_button = tk.Button(
    main_screen,
    text="Método de Secante",
    font=(font, size1),
    bg=button_bg,
    command=lambda: switch_screen(secante),
)
secante_button.pack()

multiple_roots_button = tk.Button(
    main_screen,
    text="Método de Raices multiples",
    font=(font, size1),
    bg=button_bg,
    command=lambda: switch_screen(multiple_roots),
)
multiple_roots_button.pack()

simple_gauss_button = tk.Button(
    main_screen,
    text="Método de Gauss simple",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(simple_gauss),
)
simple_gauss_button.pack()

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
matrix_entries = []
b_entries = []


def save_matrix():
    matrix_screen1.pack_forget()
    global current_screen
    current_screen.pack()


def create_matrix(n):
    global matrix_screen1, matrix_screen2, matrix_entries, b_entries, sg8_button
    sg8_button.pack_forget()
    for widget in matrix_screen2.winfo_children():
        widget.grid_forget()
    matrix_entries = []
    b_entries = []
    for row in range(n):
        entry_row = []
        for col in range(n):
            entry = tk.Entry(matrix_screen2, width=10)
            entry.grid(row=row, column=col, padx=5, pady=5)
            entry_row.append(entry)
        matrix_entries.append(entry_row)
        entry = tk.Entry(matrix_screen2, width=10)
        entry.grid(row=row, column=n, padx=60, pady=5)
        b_entries.append(entry)
    matrix_screen2.pack()
    sg8_button.pack(pady=20)


def get_matrix(screen):
    global matrix_screen1, matrix_screen2, current_screen
    matrix_screen1 = tk.Frame(screen, bg=main_bg)

    matrix1_label = tk.Label(
        matrix_screen1,
        text="Tamaño de la matriz:",
        font=(font, size),
        bg=main_bg,
    )
    matrix1_label.pack(pady=20)
    matrix1_entry = tk.Entry(matrix_screen1, font=(font, size1))
    matrix1_entry.pack()

    matrix1_button = tk.Button(
        matrix_screen1,
        text="Continuar",
        font=(font, size1),
        bg=button_bg,
        command=lambda: create_matrix(int(matrix1_entry.get())),
    )
    matrix1_button.pack(pady=20)

    matrix_screen2 = tk.Frame(screen, bg=main_bg)

    main_screen.pack_forget()
    screen.pack()

    # Show the selected screen
    current_screen = screen
    screen.pack()
    matrix_screen1.pack()


# ---- Incremental Search Screen ----
incremental_search = tk.Frame(window, bg=main_bg)

is1_title_label = tk.Label(
    incremental_search,
    text="Método de Búsqueda Incremental",
    font=(font, size2, "bold"),
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
is1_f_entry = tk.Entry(incremental_search, font=(font, size1))
is1_f_entry.pack()

is1_xinit = tk.Label(
    incremental_search,
    text="X inicial:",
    font=(font, size),
    bg=main_bg,
)
is1_xinit.pack(pady=10)
is1_xinit_entry = tk.Entry(incremental_search, font=(font, size1))
is1_xinit_entry.pack()

is1_dx = tk.Label(
    incremental_search,
    text="Derivada:",
    font=(font, size),
    bg=main_bg,
)
is1_dx.pack(pady=10)
is1_dx_entry = tk.Entry(incremental_search, font=(font, size1))
is1_dx_entry.pack()

is1_n = tk.Label(
    incremental_search,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
is1_n.pack(pady=10)
is1_n_entry = tk.Entry(incremental_search, font=(font, size1))
is1_n_entry.pack()

is1_show_button = tk.Button(
    incremental_search,
    text="Resolver",
    font=(font, size1),
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
    font=(font, size2, "bold"),
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
b2_f_entry = tk.Entry(bisection, font=(font, size1))
b2_f_entry.pack()

b2_xi = tk.Label(
    bisection,
    text="X Inferior:",
    font=(font, size),
    bg=main_bg,
)
b2_xi.pack(pady=10)
b2_xi_entry = tk.Entry(bisection, font=(font, size1))
b2_xi_entry.pack()

b2_xs = tk.Label(
    bisection,
    text="X Superior:",
    font=(font, size),
    bg=main_bg,
)
b2_xs.pack(pady=10)
b2_xs_entry = tk.Entry(bisection, font=(font, size1))
b2_xs_entry.pack()

b2_tol = tk.Label(
    bisection,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
b2_tol.pack(pady=10)
b2_tol_entry = tk.Entry(bisection, font=(font, size1))
b2_tol_entry.pack()

b2_n = tk.Label(
    bisection,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
b2_n.pack(pady=10)
b2_n_entry = tk.Entry(bisection, font=(font, size1))
b2_n_entry.pack()

b2_show_button = tk.Button(
    bisection,
    text="Resolver",
    font=(font, size1),
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
    font=(font, size2, "bold"),
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
rf3_f_entry = tk.Entry(regla_falsa, font=(font, size1))
rf3_f_entry.pack()

rf3_terror = tk.Label(
    regla_falsa,
    text="t error:",
    font=(font, size),
    bg=main_bg,
)
rf3_terror.pack(pady=10)
rf3_terror_entry = tk.Entry(regla_falsa, font=(font, size1))
rf3_terror_entry.pack()

rf3_xi = tk.Label(
    regla_falsa,
    text="X Inferior:",
    font=(font, size),
    bg=main_bg,
)
rf3_xi.pack(pady=10)
rf3_xi_entry = tk.Entry(regla_falsa, font=(font, size1))
rf3_xi_entry.pack()

rf3_xs = tk.Label(
    regla_falsa,
    text="X Superior:",
    font=(font, size),
    bg=main_bg,
)
rf3_xs.pack(pady=10)
rf3_xs_entry = tk.Entry(regla_falsa, font=(font, size1))
rf3_xs_entry.pack()

rf3_tol = tk.Label(
    regla_falsa,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
rf3_tol.pack(pady=10)
rf3_tol_entry = tk.Entry(regla_falsa, font=(font, size1))
rf3_tol_entry.pack()

rf3_n = tk.Label(
    regla_falsa,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
rf3_n.pack(pady=10)
rf3_n_entry = tk.Entry(regla_falsa, font=(font, size1))
rf3_n_entry.pack()

rf3_show_button = tk.Button(
    regla_falsa,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.regla_falsa(
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

# ---- Punto fijo Screen ----
punto_fijo = tk.Frame(window, bg=main_bg)

pf4_title_label = tk.Label(
    punto_fijo,
    text="Método del Punto fijo",
    font=(font, size2, "bold"),
    bg=main_bg,
)
pf4_title_label.pack(pady=20)

pf4_f = tk.Label(
    punto_fijo,
    text="Función F:",
    font=(font, size),
    bg=main_bg,
)
pf4_f.pack(pady=10)
pf4_f_entry = tk.Entry(punto_fijo, font=(font, size1))
pf4_f_entry.pack()

pf4_g = tk.Label(
    punto_fijo,
    text="Función G:",
    font=(font, size),
    bg=main_bg,
)
pf4_g.pack(pady=10)
pf4_g_entry = tk.Entry(punto_fijo, font=(font, size1))
pf4_g_entry.pack()

pf4_x0 = tk.Label(
    punto_fijo,
    text="X Inicial:",
    font=(font, size),
    bg=main_bg,
)
pf4_x0.pack(pady=10)
pf4_x0_entry = tk.Entry(punto_fijo, font=(font, size1))
pf4_x0_entry.pack()

pf4_tol = tk.Label(
    punto_fijo,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
pf4_tol.pack(pady=10)
pf4_tol_entry = tk.Entry(punto_fijo, font=(font, size1))
pf4_tol_entry.pack()

pf4_n = tk.Label(
    punto_fijo,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
pf4_n.pack(pady=10)
pf4_n_entry = tk.Entry(punto_fijo, font=(font, size1))
pf4_n_entry.pack()

pf4_show_button = tk.Button(
    punto_fijo,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.punto_fijo(
            pf4_f_entry.get(),
            pf4_g_entry.get(),
            pf4_x0_entry.get(),
            pf4_tol_entry.get(),
            pf4_n_entry.get(),
        )
    ),
)
pf4_show_button.pack(pady=20)

# ---- newton_raphson Screen ----
newton_raphson = tk.Frame(window, bg=main_bg)

nr5_title_label = tk.Label(
    newton_raphson,
    text="Método de Newton-Raphson",
    font=(font, size2, "bold"),
    bg=main_bg,
)
nr5_title_label.pack(pady=20)

nr5_f = tk.Label(
    newton_raphson,
    text="Función F:",
    font=(font, size),
    bg=main_bg,
)
nr5_f.pack(pady=10)
nr5_f_entry = tk.Entry(newton_raphson, font=(font, size1))
nr5_f_entry.pack()

nr5_df = tk.Label(
    newton_raphson,
    text="df:",
    font=(font, size),
    bg=main_bg,
)
nr5_df.pack(pady=10)
nr5_df_entry = tk.Entry(newton_raphson, font=(font, size1))
nr5_df_entry.pack()

nr5_x0 = tk.Label(
    newton_raphson,
    text="X Inicial:",
    font=(font, size),
    bg=main_bg,
)
nr5_x0.pack(pady=10)
nr5_x0_entry = tk.Entry(newton_raphson, font=(font, size1))
nr5_x0_entry.pack()

nr5_tol = tk.Label(
    newton_raphson,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
nr5_tol.pack(pady=10)
nr5_tol_entry = tk.Entry(newton_raphson, font=(font, size1))
nr5_tol_entry.pack()

nr5_n = tk.Label(
    newton_raphson,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
nr5_n.pack(pady=10)
nr5_n_entry = tk.Entry(newton_raphson, font=(font, size1))
nr5_n_entry.pack()

nr5_show_button = tk.Button(
    newton_raphson,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.newton_raphson(
            nr5_f_entry.get(),
            nr5_df_entry.get(),
            nr5_x0_entry.get(),
            nr5_tol_entry.get(),
            nr5_n_entry.get(),
        )
    ),
)
nr5_show_button.pack(pady=20)


# ---- secante Screen ----
secante = tk.Frame(window, bg=main_bg)

s6_title_label = tk.Label(
    secante,
    text="Método de Secante",
    font=(font, size2, "bold"),
    bg=main_bg,
)
s6_title_label.pack(pady=20)

s6_f = tk.Label(
    secante,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
s6_f.pack(pady=10)
s6_f_entry = tk.Entry(secante, font=(font, size1))
s6_f_entry.pack()

s6_x0 = tk.Label(
    secante,
    text="X0:",
    font=(font, size),
    bg=main_bg,
)
s6_x0.pack(pady=10)
s6_x0_entry = tk.Entry(secante, font=(font, size1))
s6_x0_entry.pack()

s6_x1 = tk.Label(
    secante,
    text="X1:",
    font=(font, size),
    bg=main_bg,
)
s6_x1.pack(pady=10)
s6_x1_entry = tk.Entry(secante, font=(font, size1))
s6_x1_entry.pack()

s6_tol = tk.Label(
    secante,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
s6_tol.pack(pady=10)
s6_tol_entry = tk.Entry(secante, font=(font, size1))
s6_tol_entry.pack()

s6_n = tk.Label(
    secante,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
s6_n.pack(pady=10)
s6_n_entry = tk.Entry(secante, font=(font, size1))
s6_n_entry.pack()

s6_show_button = tk.Button(
    secante,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.secante(
            s6_x0_entry.get(),
            s6_x1_entry.get(),
            s6_f_entry.get(),
            s6_tol_entry.get(),
            s6_n_entry.get(),
        )
    ),
)
s6_show_button.pack(pady=20)


# ---- multiple_roots Screen ----
multiple_roots = tk.Frame(window, bg=main_bg)

mr7_title_label = tk.Label(
    multiple_roots,
    text="Método de Reices multiples",
    font=(font, size2, "bold"),
    bg=main_bg,
)
mr7_title_label.pack(pady=20)

mr7_x0 = tk.Label(
    multiple_roots,
    text="X0:",
    font=(font, size),
    bg=main_bg,
)
mr7_x0.pack(pady=10)
mr7_x0_entry = tk.Entry(multiple_roots, font=(font, size1))
mr7_x0_entry.pack()

mr7_f = tk.Label(
    multiple_roots,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
mr7_f.pack(pady=10)
mr7_f_entry = tk.Entry(multiple_roots, font=(font, size1))
mr7_f_entry.pack()

mr7_df = tk.Label(
    multiple_roots,
    text="df:",
    font=(font, size),
    bg=main_bg,
)
mr7_df.pack(pady=10)
mr7_df_entry = tk.Entry(multiple_roots, font=(font, size1))
mr7_df_entry.pack()

mr7_df2 = tk.Label(
    multiple_roots,
    text="df2:",
    font=(font, size),
    bg=main_bg,
)
mr7_df2.pack(pady=10)
mr7_df2_entry = tk.Entry(multiple_roots, font=(font, size1))
mr7_df2_entry.pack()

mr7_tol = tk.Label(
    multiple_roots,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
mr7_tol.pack(pady=10)
mr7_tol_entry = tk.Entry(multiple_roots, font=(font, size1))
mr7_tol_entry.pack()

mr7_n = tk.Label(
    multiple_roots,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
mr7_n.pack(pady=10)
mr7_n_entry = tk.Entry(multiple_roots, font=(font, size1))
mr7_n_entry.pack()

mr7_show_button = tk.Button(
    multiple_roots,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.multiple_roots(
            mr7_x0_entry.get(),
            mr7_f_entry.get(),
            mr7_df_entry.get(),
            mr7_df2_entry.get(),
            mr7_tol_entry.get(),
            mr7_n_entry.get(),
        )
    ),
)
mr7_show_button.pack(pady=20)


# ---- simple_gauss Screen ----
simple_gauss = tk.Frame(window, bg=main_bg)

sg8_title_label = tk.Label(
    simple_gauss,
    text="Método de Gauss simple",
    font=(font, size2, "bold"),
    bg=main_bg,
)
sg8_title_label.pack(pady=20)

sg8_button = tk.Button(
    simple_gauss,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.simple_gauss(matrix_entries, b_entries)),
)

# Initially show the input screen
main_screen.pack()

# Run the application
window.mainloop()
