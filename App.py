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
    command=lambda: get_matrix(simple_gauss, sg_button),
)
simple_gauss_button.pack()

to_aug_button = tk.Button(
    main_screen,
    text="Método de to_aug",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(to_aug, ta_button),
)
to_aug_button.pack()

gauss_partial_pivot_button = tk.Button(
    main_screen,
    text="Método de Gauss de pivote parcial",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(gauss_partial_pivot, gpp_button),
)
gauss_partial_pivot_button.pack()

gauss_total_pivot_button = tk.Button(
    main_screen,
    text="Método de Gauss de pivote total",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(gauss_total_pivot, gtp_button),
)
gauss_total_pivot_button.pack()

lu_gauss_button = tk.Button(
    main_screen,
    text="Método de lu_gauss",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(lu_gauss, lug_button),
)
lu_gauss_button.pack()

LU_partial_decomposition_button = tk.Button(
    main_screen,
    text="Método de LU_partial_decomposition",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(LU_partial_decomposition, lupd_button),
)
LU_partial_decomposition_button.pack()

crout_button = tk.Button(
    main_screen,
    text="Método de Crout",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(crout, c_button),
)
crout_button.pack()

dolittle_fac = tk.Button(
    main_screen,
    text="Método de dolittle_fac",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(dolittle_fac, df_button),
)
dolittle_fac.pack()

cholesky_factorization = tk.Button(
    main_screen,
    text="Método de cholesky_factorization",
    font=(font, size1),
    bg=button_bg,
    command=lambda: get_matrix(cholesky_factorization, cf_button),
)
cholesky_factorization.pack()

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


def create_matrix(n, button):
    global matrix_screen1, matrix_screen2, matrix_entries, b_entries
    button.pack_forget()
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
    button.pack(pady=20)


def get_matrix(screen, button):
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
        command=lambda: create_matrix(int(matrix1_entry.get()), button),
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

is_title_label = tk.Label(
    incremental_search,
    text="Método de Búsqueda Incremental",
    font=(font, size2, "bold"),
    bg=main_bg,
)
is_title_label.pack(pady=20)

is_f = tk.Label(
    incremental_search,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
is_f.pack(pady=10)
is_f_entry = tk.Entry(incremental_search, font=(font, size1))
is_f_entry.pack()

is_xinit = tk.Label(
    incremental_search,
    text="X inicial:",
    font=(font, size),
    bg=main_bg,
)
is_xinit.pack(pady=10)
is_xinit_entry = tk.Entry(incremental_search, font=(font, size1))
is_xinit_entry.pack()

is_dx = tk.Label(
    incremental_search,
    text="Derivada:",
    font=(font, size),
    bg=main_bg,
)
is_dx.pack(pady=10)
is_dx_entry = tk.Entry(incremental_search, font=(font, size1))
is_dx_entry.pack()

is_n = tk.Label(
    incremental_search,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
is_n.pack(pady=10)
is_n_entry = tk.Entry(incremental_search, font=(font, size1))
is_n_entry.pack()

is_show_button = tk.Button(
    incremental_search,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.incremental_search(
            is_f_entry.get(),
            is_xinit_entry.get(),
            is_dx_entry.get(),
            is_n_entry.get(),
        )
    ),
)
is_show_button.pack(pady=20)

# ---- Bisection Screen ----
bisection = tk.Frame(window, bg=main_bg)

b_title_label = tk.Label(
    bisection,
    text="Método de Bisección",
    font=(font, size2, "bold"),
    bg=main_bg,
)
b_title_label.pack(pady=20)

b_f = tk.Label(
    bisection,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
b_f.pack(pady=10)
b_f_entry = tk.Entry(bisection, font=(font, size1))
b_f_entry.pack()

b_xi = tk.Label(
    bisection,
    text="X Inferior:",
    font=(font, size),
    bg=main_bg,
)
b_xi.pack(pady=10)
b_xi_entry = tk.Entry(bisection, font=(font, size1))
b_xi_entry.pack()

b_xs = tk.Label(
    bisection,
    text="X Superior:",
    font=(font, size),
    bg=main_bg,
)
b_xs.pack(pady=10)
b_xs_entry = tk.Entry(bisection, font=(font, size1))
b_xs_entry.pack()

b_tol = tk.Label(
    bisection,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
b_tol.pack(pady=10)
b_tol_entry = tk.Entry(bisection, font=(font, size1))
b_tol_entry.pack()

b_n = tk.Label(
    bisection,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
b_n.pack(pady=10)
b_n_entry = tk.Entry(bisection, font=(font, size1))
b_n_entry.pack()

b_show_button = tk.Button(
    bisection,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.biseccion(
            b_f_entry.get(),
            b_xi_entry.get(),
            b_xs_entry.get(),
            b_tol_entry.get(),
            b_n_entry.get(),
        )
    ),
)
b_show_button.pack(pady=20)

# ---- regla_falsa Screen ----
regla_falsa = tk.Frame(window, bg=main_bg)

rf_title_label = tk.Label(
    regla_falsa,
    text="Método de la Regla Falsa",
    font=(font, size2, "bold"),
    bg=main_bg,
)
rf_title_label.pack(pady=20)

rf_f = tk.Label(
    regla_falsa,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
rf_f.pack(pady=10)
rf_f_entry = tk.Entry(regla_falsa, font=(font, size1))
rf_f_entry.pack()

rf_terror = tk.Label(
    regla_falsa,
    text="t error:",
    font=(font, size),
    bg=main_bg,
)
rf_terror.pack(pady=10)
rf_terror_entry = tk.Entry(regla_falsa, font=(font, size1))
rf_terror_entry.pack()

rf_xi = tk.Label(
    regla_falsa,
    text="X Inferior:",
    font=(font, size),
    bg=main_bg,
)
rf_xi.pack(pady=10)
rf_xi_entry = tk.Entry(regla_falsa, font=(font, size1))
rf_xi_entry.pack()

rf_xs = tk.Label(
    regla_falsa,
    text="X Superior:",
    font=(font, size),
    bg=main_bg,
)
rf_xs.pack(pady=10)
rf_xs_entry = tk.Entry(regla_falsa, font=(font, size1))
rf_xs_entry.pack()

rf_tol = tk.Label(
    regla_falsa,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
rf_tol.pack(pady=10)
rf_tol_entry = tk.Entry(regla_falsa, font=(font, size1))
rf_tol_entry.pack()

rf_n = tk.Label(
    regla_falsa,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
rf_n.pack(pady=10)
rf_n_entry = tk.Entry(regla_falsa, font=(font, size1))
rf_n_entry.pack()

rf_show_button = tk.Button(
    regla_falsa,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.regla_falsa(
            rf_f_entry.get(),
            rf_terror_entry.get(),
            rf_xi_entry.get(),
            rf_xs_entry.get(),
            rf_tol_entry.get(),
            rf_n_entry.get(),
        )
    ),
)
rf_show_button.pack(pady=20)

# ---- Punto fijo Screen ----
punto_fijo = tk.Frame(window, bg=main_bg)

pf_title_label = tk.Label(
    punto_fijo,
    text="Método del Punto fijo",
    font=(font, size2, "bold"),
    bg=main_bg,
)
pf_title_label.pack(pady=20)

pf_f = tk.Label(
    punto_fijo,
    text="Función F:",
    font=(font, size),
    bg=main_bg,
)
pf_f.pack(pady=10)
pf_f_entry = tk.Entry(punto_fijo, font=(font, size1))
pf_f_entry.pack()

pf_g = tk.Label(
    punto_fijo,
    text="Función G:",
    font=(font, size),
    bg=main_bg,
)
pf_g.pack(pady=10)
pf_g_entry = tk.Entry(punto_fijo, font=(font, size1))
pf_g_entry.pack()

pf_x0 = tk.Label(
    punto_fijo,
    text="X Inicial:",
    font=(font, size),
    bg=main_bg,
)
pf_x0.pack(pady=10)
pf_x0_entry = tk.Entry(punto_fijo, font=(font, size1))
pf_x0_entry.pack()

pf_tol = tk.Label(
    punto_fijo,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
pf_tol.pack(pady=10)
pf_tol_entry = tk.Entry(punto_fijo, font=(font, size1))
pf_tol_entry.pack()

pf_n = tk.Label(
    punto_fijo,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
pf_n.pack(pady=10)
pf_n_entry = tk.Entry(punto_fijo, font=(font, size1))
pf_n_entry.pack()

pf_show_button = tk.Button(
    punto_fijo,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.punto_fijo(
            pf_f_entry.get(),
            pf_g_entry.get(),
            pf_x0_entry.get(),
            pf_tol_entry.get(),
            pf_n_entry.get(),
        )
    ),
)
pf_show_button.pack(pady=20)

# ---- newton_raphson Screen ----
newton_raphson = tk.Frame(window, bg=main_bg)

nr_title_label = tk.Label(
    newton_raphson,
    text="Método de Newton-Raphson",
    font=(font, size2, "bold"),
    bg=main_bg,
)
nr_title_label.pack(pady=20)

nr_f = tk.Label(
    newton_raphson,
    text="Función F:",
    font=(font, size),
    bg=main_bg,
)
nr_f.pack(pady=10)
nr_f_entry = tk.Entry(newton_raphson, font=(font, size1))
nr_f_entry.pack()

nr_df = tk.Label(
    newton_raphson,
    text="df:",
    font=(font, size),
    bg=main_bg,
)
nr_df.pack(pady=10)
nr_df_entry = tk.Entry(newton_raphson, font=(font, size1))
nr_df_entry.pack()

nr_x0 = tk.Label(
    newton_raphson,
    text="X Inicial:",
    font=(font, size),
    bg=main_bg,
)
nr_x0.pack(pady=10)
nr_x0_entry = tk.Entry(newton_raphson, font=(font, size1))
nr_x0_entry.pack()

nr_tol = tk.Label(
    newton_raphson,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
nr_tol.pack(pady=10)
nr_tol_entry = tk.Entry(newton_raphson, font=(font, size1))
nr_tol_entry.pack()

nr_n = tk.Label(
    newton_raphson,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
nr_n.pack(pady=10)
nr_n_entry = tk.Entry(newton_raphson, font=(font, size1))
nr_n_entry.pack()

nr_show_button = tk.Button(
    newton_raphson,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.newton_raphson(
            nr_f_entry.get(),
            nr_df_entry.get(),
            nr_x0_entry.get(),
            nr_tol_entry.get(),
            nr_n_entry.get(),
        )
    ),
)
nr_show_button.pack(pady=20)


# ---- secante Screen ----
secante = tk.Frame(window, bg=main_bg)

s_title_label = tk.Label(
    secante,
    text="Método de Secante",
    font=(font, size2, "bold"),
    bg=main_bg,
)
s_title_label.pack(pady=20)

s_f = tk.Label(
    secante,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
s_f.pack(pady=10)
s_f_entry = tk.Entry(secante, font=(font, size1))
s_f_entry.pack()

s_x0 = tk.Label(
    secante,
    text="X0:",
    font=(font, size),
    bg=main_bg,
)
s_x0.pack(pady=10)
s_x0_entry = tk.Entry(secante, font=(font, size1))
s_x0_entry.pack()

s_x1 = tk.Label(
    secante,
    text="X1:",
    font=(font, size),
    bg=main_bg,
)
s_x1.pack(pady=10)
s_x1_entry = tk.Entry(secante, font=(font, size1))
s_x1_entry.pack()

s_tol = tk.Label(
    secante,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
s_tol.pack(pady=10)
s_tol_entry = tk.Entry(secante, font=(font, size1))
s_tol_entry.pack()

s_n = tk.Label(
    secante,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
s_n.pack(pady=10)
s_n_entry = tk.Entry(secante, font=(font, size1))
s_n_entry.pack()

s_show_button = tk.Button(
    secante,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.secante(
            s_x0_entry.get(),
            s_x1_entry.get(),
            s_f_entry.get(),
            s_tol_entry.get(),
            s_n_entry.get(),
        )
    ),
)
s_show_button.pack(pady=20)


# ---- multiple_roots Screen ----
multiple_roots = tk.Frame(window, bg=main_bg)

mr_title_label = tk.Label(
    multiple_roots,
    text="Método de Reices multiples",
    font=(font, size2, "bold"),
    bg=main_bg,
)
mr_title_label.pack(pady=20)

mr_x0 = tk.Label(
    multiple_roots,
    text="X0:",
    font=(font, size),
    bg=main_bg,
)
mr_x0.pack(pady=10)
mr_x0_entry = tk.Entry(multiple_roots, font=(font, size1))
mr_x0_entry.pack()

mr_f = tk.Label(
    multiple_roots,
    text="Función:",
    font=(font, size),
    bg=main_bg,
)
mr_f.pack(pady=10)
mr_f_entry = tk.Entry(multiple_roots, font=(font, size1))
mr_f_entry.pack()

mr_df = tk.Label(
    multiple_roots,
    text="df:",
    font=(font, size),
    bg=main_bg,
)
mr_df.pack(pady=10)
mr_df_entry = tk.Entry(multiple_roots, font=(font, size1))
mr_df_entry.pack()

mr_df2 = tk.Label(
    multiple_roots,
    text="df2:",
    font=(font, size),
    bg=main_bg,
)
mr_df2.pack(pady=10)
mr_df2_entry = tk.Entry(multiple_roots, font=(font, size1))
mr_df2_entry.pack()

mr_tol = tk.Label(
    multiple_roots,
    text="Tolerancia:",
    font=(font, size),
    bg=main_bg,
)
mr_tol.pack(pady=10)
mr_tol_entry = tk.Entry(multiple_roots, font=(font, size1))
mr_tol_entry.pack()

mr_n = tk.Label(
    multiple_roots,
    text="Iteraciones:",
    font=(font, size),
    bg=main_bg,
)
mr_n.pack(pady=10)
mr_n_entry = tk.Entry(multiple_roots, font=(font, size1))
mr_n_entry.pack()

mr_show_button = tk.Button(
    multiple_roots,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.multiple_roots(
            mr_x0_entry.get(),
            mr_f_entry.get(),
            mr_df_entry.get(),
            mr_df2_entry.get(),
            mr_tol_entry.get(),
            mr_n_entry.get(),
        )
    ),
)
mr_show_button.pack(pady=20)


# ---- simple_gauss Screen ----
simple_gauss = tk.Frame(window, bg=main_bg)

sg_title_label = tk.Label(
    simple_gauss,
    text="Método de Gauss simple",
    font=(font, size2, "bold"),
    bg=main_bg,
)
sg_title_label.pack(pady=20)

sg_button = tk.Button(
    simple_gauss,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.simple_gauss(matrix_entries, b_entries)),
)

# ---- to_aug Screen ----
to_aug = tk.Frame(window, bg=main_bg)

ta_title_label = tk.Label(
    to_aug,
    text="Método de to_aug",
    font=(font, size2, "bold"),
    bg=main_bg,
)
ta_title_label.pack(pady=20)

ta_button = tk.Button(
    to_aug,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.to_aug(matrix_entries, b_entries)),
)

# ---- gauss_partial_pivot Screen ----
gauss_partial_pivot = tk.Frame(window, bg=main_bg)

gpp_title_label = tk.Label(
    gauss_partial_pivot,
    text="Método de Gauss de pivote parcial",
    font=(font, size2, "bold"),
    bg=main_bg,
)
gpp_title_label.pack(pady=20)

gpp_button = tk.Button(
    gauss_partial_pivot,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.gauss_partial_pivot(matrix_entries, b_entries)),
)

# ---- gauss_total_pivot Screen ----
gauss_total_pivot = tk.Frame(window, bg=main_bg)

gtp_title_label = tk.Label(
    gauss_total_pivot,
    text="Método de Gauss de pivote total",
    font=(font, size2, "bold"),
    bg=main_bg,
)
gtp_title_label.pack(pady=20)

gtp_button = tk.Button(
    gauss_total_pivot,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.gauss_total_pivot(matrix_entries, b_entries)),
)

# ---- lu_gauss Screen ----
lu_gauss = tk.Frame(window, bg=main_bg)

lug_title_label = tk.Label(
    lu_gauss,
    text="Método de lu_gauss",
    font=(font, size2, "bold"),
    bg=main_bg,
)
lug_title_label.pack(pady=20)

lug_button = tk.Button(
    lu_gauss,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.lu_gauss(matrix_entries, b_entries)),
)

# ---- LU_partial_decomposition Screen ----
LU_partial_decomposition = tk.Frame(window, bg=main_bg)

lupd_title_label = tk.Label(
    lu_gauss,
    text="Método de LU_partial_decomposition",
    font=(font, size2, "bold"),
    bg=main_bg,
)
lupd_title_label.pack(pady=20)

lupd_button = tk.Button(
    LU_partial_decomposition,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.LU_partial_decomposition(matrix_entries, b_entries)),
)

# ---- crout Screen ----
crout = tk.Frame(window, bg=main_bg)

c_title_label = tk.Label(
    lu_gauss,
    text="Método de Crout",
    font=(font, size2, "bold"),
    bg=main_bg,
)
c_title_label.pack(pady=20)

c_button = tk.Button(
    crout,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.crout(matrix_entries, b_entries)),
)

# ---- dolittle_fac Screen ----
dolittle_fac = tk.Frame(window, bg=main_bg)

df_title_label = tk.Label(
    lu_gauss,
    text="Método de dolittle_fac",
    font=(font, size2, "bold"),
    bg=main_bg,
)
df_title_label.pack(pady=20)

df_button = tk.Button(
    dolittle_fac,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.dolittle_fac(matrix_entries, b_entries)),
)

# ---- cholesky_factorization Screen ----
cholesky_factorization = tk.Frame(window, bg=main_bg)

cf_title_label = tk.Label(
    lu_gauss,
    text="Método de cholesky_factorization",
    font=(font, size2, "bold"),
    bg=main_bg,
)
cf_title_label.pack(pady=20)

cf_button = tk.Button(
    cholesky_factorization,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.cholesky_factorization(matrix_entries, b_entries)),
)

# Initially show the input screen
main_screen.pack()

# Run the application
window.mainloop()
