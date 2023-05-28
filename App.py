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
button_width = 30


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
    matrix_screen1.pack_forget()
    matrix_screen2.pack_forget()
    current_screen = main_screen

    # Show the main screen
    main_screen.pack()


# Create a new themed window
window = ThemedTk(theme="radiant")
window.title("Métodos Análisis Numérico")
window.configure(background=main_bg)
window.state("zoomed")
window.geometry("800x500")


# ---- Matrix Screen ----
matrix_entries = []
matb_entries = []
matrix_screen1 = tk.Frame(window, bg=main_bg)
matrix_screen2 = tk.Frame(window, bg=main_bg)


def create_matrix(n, button):
    global matrix_screen1, matrix_screen2, matrix_entries, matb_entries
    for widget in matrix_screen2.winfo_children():
        widget.grid_forget()
    matrix_entries = []
    matb_entries = []
    for row in range(n):
        entry_row = []
        for col in range(n):
            entry = tk.Entry(matrix_screen2, width=10)
            entry.grid(row=row, column=col, padx=5, pady=5)
            entry_row.append(entry)
        matrix_entries.append(entry_row)
        entry = tk.Entry(matrix_screen2, width=10)
        entry.grid(row=row, column=n, padx=60, pady=5)
        matb_entries.append(entry)
    matrix_screen2.pack()
    button.pack(pady=20)


def get_matrix(screen, button):
    global matrix_screen1, matrix_screen2, current_screen
    matrix_screen1 = tk.Frame(screen, bg=main_bg)
    button.pack_forget()

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


# ---- Incremental Search Screen ----
incremental_search = tk.Frame(window, bg=main_bg)

tk.Label(
    incremental_search,
    text="Método de Búsqueda Incremental",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

is_labels = ["Función:", "X inicial:", "Derivada:", "Iteraciones:"]
is_entries = [None] * len(is_labels)
count = 0

for label_text in is_labels:
    tk.Label(
        incremental_search,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    is_entries[count] = tk.Entry(incremental_search, font=(font, size1))
    is_entries[count].pack()
    count += 1


is_show_button = tk.Button(
    incremental_search,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.incremental_search(
            is_entries[0].get(),
            is_entries[1].get(),
            is_entries[2].get(),
            is_entries[3].get(),
        )
    ),
)
is_show_button.pack(pady=20)

# ---- Bisection Screen ----
bisection = tk.Frame(window, bg=main_bg)

tk.Label(
    bisection,
    text="Método de Bisección",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

b_labels = ["Función:", "X Inferior:", "X Superior:", "Tolerancia:", "Iteraciones:"]
b_entries = [None] * len(b_labels)
count = 0

for label_text in b_labels:
    tk.Label(
        bisection,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    b_entries[count] = tk.Entry(bisection, font=(font, size1))
    b_entries[count].pack()
    count += 1

b_show_button = tk.Button(
    bisection,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.biseccion(
            b_entries[0].get(),
            b_entries[1].get(),
            b_entries[2].get(),
            b_entries[3].get(),
            b_entries[4].get(),
        )
    ),
)
b_show_button.pack(pady=20)

# ---- regla_falsa Screen ----
regla_falsa = tk.Frame(window, bg=main_bg)

tk.Label(
    regla_falsa,
    text="Método de la Regla Falsa",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

rf_labels = [
    "Función:",
    "t error:",
    "X Inferior:",
    "X Superior:",
    "Tolerancia:",
    "Iteraciones:",
]
rf_entries = [None] * len(rf_labels)
count = 0

for label_text in rf_labels:
    tk.Label(
        regla_falsa,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    rf_entries[count] = tk.Entry(regla_falsa, font=(font, size1))
    rf_entries[count].pack()
    count += 1

rf_show_button = tk.Button(
    regla_falsa,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.regla_falsa(
            rf_entries[0].get(),
            rf_entries[1].get(),
            rf_entries[2].get(),
            rf_entries[3].get(),
            rf_entries[4].get(),
            rf_entries[5].get(),
        )
    ),
)
rf_show_button.pack(pady=20)

# ---- Punto fijo Screen ----
punto_fijo = tk.Frame(window, bg=main_bg)

tk.Label(
    punto_fijo,
    text="Método del Punto fijo",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

pf_labels = ["Función F:", "Función F:", "X Inicial:", "Tolerancia:", "Iteraciones:"]
pf_entries = [None] * len(pf_labels)
count = 0

for label_text in pf_labels:
    tk.Label(
        punto_fijo,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    pf_entries[count] = tk.Entry(punto_fijo, font=(font, size1))
    pf_entries[count].pack()


pf_show_button = tk.Button(
    punto_fijo,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.punto_fijo(
            pf_entries[0].get(),
            pf_entries[1].get(),
            pf_entries[2].get(),
            pf_entries[3].get(),
            pf_entries[4].get(),
        )
    ),
)
pf_show_button.pack(pady=20)

# ---- newton_raphson Screen ----
newton_raphson = tk.Frame(window, bg=main_bg)

tk.Label(
    newton_raphson,
    text="Método de Newton-Raphson",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

nr_labels = ["Función:", "df:", "X Inicial:", "Tolerancia:", "Iteraciones:"]
nr_entries = [None] * len(nr_labels)
count = 0

for label_text in nr_labels:
    tk.Label(
        newton_raphson,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    nr_entries[count] = tk.Entry(newton_raphson, font=(font, size1))
    nr_entries[count].pack()

nr_show_button = tk.Button(
    newton_raphson,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.newton_raphson(
            nr_entries[0].get(),
            nr_entries[1].get(),
            nr_entries[2].get(),
            nr_entries[3].get(),
            nr_entries[4].get(),
        )
    ),
)
nr_show_button.pack(pady=20)


# ---- secante Screen ----
secante = tk.Frame(window, bg=main_bg)

tk.Label(
    secante,
    text="Método de Secante",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

s_labels = ["Función:", "X0:", "X1:", "Tolerancia:", "Iteraciones:"]
s_entries = [None] * len(s_labels)
count = 0

for label_text in s_labels:
    tk.Label(
        secante,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    s_entries[count] = tk.Entry(secante, font=(font, size1))
    s_entries[count].pack()

s_show_button = tk.Button(
    secante,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.secante(
            s_entries[1].get(),
            s_entries[2].get(),
            s_entries[0].get(),
            s_entries[3].get(),
            s_entries[4].get(),
        )
    ),
)
s_show_button.pack(pady=20)


# ---- multiple_roots Screen ----
multiple_roots = tk.Frame(window, bg=main_bg)

tk.Label(
    multiple_roots,
    text="Método de Reices multiples",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

mr_labels = ["X0", "Función:", "df:", "df2:", "Tolerancia:", "Iteraciones:"]
mr_entries = [None] * len(mr_labels)
count = 0

for label_text in mr_labels:
    tk.Label(
        multiple_roots,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    mr_entries[count] = tk.Entry(multiple_roots, font=(font, size1))
    mr_entries[count].pack()

mr_show_button = tk.Button(
    multiple_roots,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.multiple_roots(
            mr_entries[0].get(),
            mr_entries[1].get(),
            mr_entries[2].get(),
            mr_entries[3].get(),
            mr_entries[4].get(),
            mr_entries[5].get(),
        )
    ),
)
mr_show_button.pack(pady=20)


# ---- simple_gauss Screen ----
simple_gauss = tk.Frame(window, bg=main_bg)

tk.Label(
    simple_gauss,
    text="Método de Gauss simple",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

sg_button = tk.Button(
    simple_gauss,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.simple_gauss(matrix_entries, matb_entries)),
)

# ---- to_aug Screen ----
to_aug = tk.Frame(window, bg=main_bg)

tk.Label(
    to_aug,
    text="Método de to_aug",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

ta_button = tk.Button(
    to_aug,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.to_aug(matrix_entries, matb_entries)),
)

# ---- gauss_partial_pivot Screen ----
gauss_partial_pivot = tk.Frame(window, bg=main_bg)

tk.Label(
    gauss_partial_pivot,
    text="Método de Gauss de pivote parcial",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

gpp_button = tk.Button(
    gauss_partial_pivot,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.gauss_partial_pivot(matrix_entries, matb_entries)
    ),
)

# ---- gauss_total_pivot Screen ----
gauss_total_pivot = tk.Frame(window, bg=main_bg)

tk.Label(
    gauss_total_pivot,
    text="Método de Gauss de pivote total",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

gtp_button = tk.Button(
    gauss_total_pivot,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.gauss_total_pivot(matrix_entries, matb_entries)
    ),
)

# ---- lu_gauss Screen ----
lu_gauss = tk.Frame(window, bg=main_bg)

tk.Label(
    lu_gauss,
    text="Método de lu_gauss",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

lug_button = tk.Button(
    lu_gauss,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.lu_gauss(matrix_entries, matb_entries)),
)

# ---- LU_partial_decomposition Screen ----
LU_partial_decomposition = tk.Frame(window, bg=main_bg)

tk.Label(
    LU_partial_decomposition,
    text="Método de LU_partial_decomposition",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

lupd_button = tk.Button(
    LU_partial_decomposition,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.LU_partial_decomposition(matrix_entries, matb_entries)
    ),
)

# ---- crout Screen ----
crout = tk.Frame(window, bg=main_bg)

tk.Label(
    crout,
    text="Método de Crout",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

c_button = tk.Button(
    crout,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.crout(matrix_entries, matb_entries)),
)

# ---- dolittle_fac Screen ----
dolittle_fac = tk.Frame(window, bg=main_bg)

tk.Label(
    dolittle_fac,
    text="Método de dolittle_fac",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

df_button = tk.Button(
    dolittle_fac,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(methods.dolittle_fac(matrix_entries, matb_entries)),
)

# ---- cholesky_factorization Screen ----
cholesky_factorization = tk.Frame(window, bg=main_bg)

tk.Label(
    cholesky_factorization,
    text="Método de cholesky_factorization",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

cf_button = tk.Button(
    cholesky_factorization,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.cholesky_factorization(matrix_entries, matb_entries)
    ),
)

# Define button information
button_info = [
    ("Método de Búsqueda Incremental", switch_screen, incremental_search, None),
    ("Método de Bisección", switch_screen, bisection, None),
    ("Método de la Regla Falsa", switch_screen, regla_falsa, None),
    ("Método del Punto fijo", switch_screen, punto_fijo, None),
    ("Método de Newton-Raphson", switch_screen, newton_raphson, None),
    ("Método de Secante", switch_screen, secante, None),
    ("Método de Raices multiples", switch_screen, multiple_roots, None),
    ("Método de Gauss simple", get_matrix, simple_gauss, sg_button),
    ("Método de to_aug", get_matrix, to_aug, ta_button),
    ("Método de Gauss de pivote parcial", get_matrix, gauss_partial_pivot, gpp_button),
    ("Método de Gauss de pivote total", get_matrix, gauss_total_pivot, gtp_button),
    ("Método de lu_gauss", get_matrix, lu_gauss, lug_button),
    ("Método de LU_partial_decomp", get_matrix, LU_partial_decomposition, lupd_button),
    ("Método de Crout", get_matrix, crout, c_button),
    ("Método de dolittle_fac", get_matrix, dolittle_fac, df_button),
    ("Método de cholesky_factorization", get_matrix, cholesky_factorization, cf_button),
]

# ---- Main Screen ----
main_screen = tk.Frame(window, bg=main_bg)
current_screen = main_screen

title_label = tk.Label(
    main_screen,
    text="Métodos de Análisis Numérico",
    font=(font, size2, "bold"),
    bg=main_bg,
)
title_label.grid(row=0, column=1, padx=5, pady=20)

# Create and position buttons dynamically
row = 1
col = 0
for text, com, screen, button in button_info:
    if com == switch_screen:
        tk.Button(
            main_screen,
            text=text,
            font=(font, size1),
            bg=button_bg,
            width=button_width,
            command=lambda screen=screen: switch_screen(screen),
        ).grid(row=row, column=col, padx=5, pady=5)
    else:
        tk.Button(
            main_screen,
            text=text,
            font=(font, size1),
            bg=button_bg,
            width=button_width,
            command=lambda screen=screen, button=button: get_matrix(screen, button),
        ).grid(row=row, column=col, padx=5, pady=5)

    row += 1
    if row > 10:
        row = 1
        col += 1


# Initially show the input screen
main_screen.pack()

# Run the application
window.mainloop()
