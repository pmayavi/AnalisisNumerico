import tkinter as tk
from ttkthemes import ThemedTk
import metodos as methods
import numpy as np

# Style variables
font = "Arial"
size = 16
size1 = 14
size2 = 20
main_bg = "#%02x%02x%02x" % (255, 255, 151)
button_bg = "#%02x%02x%02x" % (int(255 * 0.9), int(255 * 0.9), int(151 * 0.9))
sidebar_bg = "#%02x%02x%02x" % (255, 179, 71)
sidebutton_bg = "#%02x%02x%02x" % (int(255 * 0.9), int(179 * 0.9), int(71 * 0.9))
table_bg1 = "#%02x%02x%02x" % (255, 195, 102)
table_bg2 = "#%02x%02x%02x" % (255, 189, 90)
button_width = 30


def switch_screen(screen):
    # Hide the current screen
    global current_screen
    current_screen.pack_forget()

    # Show the selected screen
    current_screen = screen
    current_screen.pack()


def show_result(method, inputs):
    global current_screen, matrix_screen1, matrix_screen2, table_screen
    current_screen.pack_forget()
    matrix_screen1.pack_forget()
    matrix_screen2.pack_forget()
    points_screen.pack_forget()
    for widget in table_screen.winfo_children():
        widget.grid_forget()

    try:
        res = method(*inputs)
    except Exception as e:
        res = str(e) + "\n" + str(inputs)
        print(e)
    print(res)

    try:
        if isinstance(res, tuple):
            if isinstance(res[1], list):
                if isinstance(res[0], np.ndarray):
                    result_label.config(
                        text="\n".join(str(item) for item in res[0].tolist())
                    )
                else:
                    result_label.config(text=" ".join(str(item) for item in res[1]))
                    show_table(res[0][-10:])
            elif isinstance(res[0][0][0], np.ndarray):
                cont = 0
                for row in res[0][-2:]:
                    for col in row:
                        show_table(col.tolist(), cont)
                        cont += len(col.tolist()) + 1
                        tk.Label(
                            table_screen,
                            bg=main_bg,
                        ).grid(row=cont - 1, column=0)
            else:
                result_label.config(text=res[1])
                show_table(res[0][-10:])
        elif isinstance(res, list):
            if isinstance(res[0], list):
                cont = 0
                for row in res:
                    show_table(row, cont)
                    cont += len(row) + 1
                    tk.Label(
                        table_screen,
                        bg=main_bg,
                    ).grid(row=cont - 1, column=0)
            else:
                result_label.config(text="\n".join(str(item) for item in res))
        else:
            result_label.config(text=str(res))
    except:
        result_label.config(
            text="\n".join(str(res)[i : i + 100] for i in range(0, len(str(res)), 100))
        )
    result_screen.pack()


def show_table(table, row=0):
    c1 = main_bg
    c2 = table_bg1
    for iter in table:
        col = 0
        color = c1
        c1 = c2
        c2 = color
        for element in iter:
            if isinstance(element, str):
                element = float(element)
            tk.Label(
                table_screen,
                text=str(round(element, 6)),
                font=(font, size),
                relief=tk.RIDGE,
                bg=color,
                borderwidth=10,
                highlightbackground=table_bg2,
                width=10,
            ).grid(row=row, column=col, sticky="nsew")
            col += 1
        row += 1


def return_to_main():
    global current_screen
    # Hide the current screen
    current_screen.pack_forget()
    matrix_screen1.pack_forget()
    matrix_screen2.pack_forget()
    result_screen.pack_forget()
    points_screen.pack_forget()
    current_screen = main_screen

    # Show the main screen
    main_screen.pack()


def get_matrix_values():
    global matrix_entries
    val = []
    for row in matrix_entries:
        arr = []
        for entry in row:
            arr.append(float(entry.get()))
        val.append(arr)
    return np.array(val)


def get_b_values():
    global matb_entries
    val = []
    for entry in matb_entries:
        val.append(float(entry.get()))
    return np.array(val)


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
    set_default_values(n)


matrix1_entry = tk.Entry(matrix_screen1, font=(font, size1))


def get_matrix(screen, button):
    global matrix_screen1, matrix_screen2, current_screen, matrix1_entry
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
    set_default_values()

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

    # Show the selected screen
    current_screen = screen
    screen.pack()
    matrix_screen1.pack()


# ---- Points Screen ----
pointsx_entries = []
pointsy_entries = []
points_screen = tk.Frame(window, bg=main_bg)


def add_row():
    global pointsx_entries, pointsy_entries
    entry = tk.Entry(points_screen, width=10)
    entry.grid(row=len(pointsx_entries) + 1, column=0, pady=10)
    pointsx_entries.append(entry)

    entry = tk.Entry(points_screen, width=10)
    entry.grid(row=len(pointsy_entries) + 1, column=1, pady=10)
    pointsy_entries.append(entry)


def create_points(screen, button):
    global current_screen, points_screen, pointsx_entries, pointsy_entries
    pointsx_entries = []
    pointsy_entries = []
    points_screen = tk.Frame(screen, bg=main_bg)
    current_screen = screen
    button.pack_forget()
    for widget in points_screen.winfo_children():
        widget.grid_forget()
    tk.Button(points_screen, text="+", command=add_row, width=10).grid(
        row=0, column=0, pady=20
    )
    main_screen.pack_forget()
    screen.pack()
    add_row()
    set_default_values()
    points_screen.pack()
    button.pack(pady=20)


def get_x_values():
    val = []
    for x in pointsx_entries:
        val.append(float(x.get()))
    return np.array(val)


def get_y_values():
    val = []
    for y in pointsy_entries:
        val.append(float(y.get()))
    return np.array(val)


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


def toggle_default():
    global defaults
    defaults = not defaults
    set_default_values()


defaults = False  # Initial value
check_button = tk.Checkbutton(
    sidebar, text="Variables default?", command=toggle_default, bg=sidebar_bg
)
check_button.pack(pady=5)

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
table_screen = tk.Frame(result_screen, bg=main_bg)
table_screen.pack()


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
        methods.incremental_search,
        (
            is_entries[0].get().replace("^", "**"),
            int(is_entries[1].get()),
            float(is_entries[2].get()),
            int(is_entries[3].get()),
        ),
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
        methods.biseccion,
        (
            b_entries[0].get().replace("^", "**"),
            float(b_entries[1].get()),
            float(b_entries[2].get()),
            float(b_entries[3].get()),
            float(b_entries[4].get()),
        ),
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
    "X Inferior:",
    "X Superior:",
    "Tolerancia:",
    "Iteraciones:",
    "t error:",
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
        methods.regla_falsa,
        (
            rf_entries[0].get().replace("^", "**"),
            float(rf_entries[1].get()),
            float(rf_entries[2].get()),
            float(rf_entries[3].get()),
            float(rf_entries[4].get()),
            float(rf_entries[5].get()),
        ),
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

pf_labels = ["Función F:", "Función G:", "X Inicial:", "Tolerancia:", "Iteraciones:"]
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
    count += 1


pf_show_button = tk.Button(
    punto_fijo,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.fixed_point,
        (
            pf_entries[0].get().replace("^", "**"),
            pf_entries[1].get().replace("^", "**"),
            float(pf_entries[2].get()),
            float(pf_entries[3].get()),
            float(pf_entries[4].get()),
        ),
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
    count += 1

nr_show_button = tk.Button(
    newton_raphson,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.newton_raphson,
        (
            nr_entries[0].get().replace("^", "**"),
            nr_entries[1].get().replace("^", "**"),
            float(nr_entries[2].get()),
            float(nr_entries[3].get()),
            float(nr_entries[4].get()),
        ),
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
    count += 1

s_show_button = tk.Button(
    secante,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.secante,
        (
            s_entries[0].get().replace("^", "**"),
            float(s_entries[1].get()),
            float(s_entries[2].get()),
            float(s_entries[3].get()),
            float(s_entries[4].get()),
        ),
    ),
)
s_show_button.pack(pady=20)


# ---- multiple_roots Screen ----
multiple_roots = tk.Frame(window, bg=main_bg)

tk.Label(
    multiple_roots,
    text="Método de Raices multiples",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

mr_labels = ["Función:", "df:", "df2:", "X0", "Tolerancia:", "Iteraciones:"]
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
    count += 1

mr_show_button = tk.Button(
    multiple_roots,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.multiple_roots,
        (
            mr_entries[0].get().replace("^", "**"),
            mr_entries[1].get().replace("^", "**"),
            mr_entries[2].get().replace("^", "**"),
            float(mr_entries[3].get()),
            float(mr_entries[4].get()),
            float(mr_entries[5].get()),
        ),
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
    command=lambda: show_result(
        methods.simple_gauss, (get_matrix_values(), get_b_values())
    ),
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
        methods.gauss_partial_pivot, (get_matrix_values(), get_b_values())
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
        methods.gauss_total_pivot, (get_matrix_values(), get_b_values())
    ),
)

# ---- lu_gauss Screen ----
lu_gauss = tk.Frame(window, bg=main_bg)

tk.Label(
    lu_gauss,
    text="Método de LU de Gauss",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

lug_button = tk.Button(
    lu_gauss,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.lu_gauss, (get_matrix_values(), get_b_values())
    ),
)

# ---- LU_partial_decomposition Screen ----
LU_partial_decomposition = tk.Frame(window, bg=main_bg)

tk.Label(
    LU_partial_decomposition,
    text="Método de Lu Descomposicion parcial",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

lupd_button = tk.Button(
    LU_partial_decomposition,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.LU_partial_decomposition, (get_matrix_values(), get_b_values())
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
    command=lambda: show_result(methods.crout, (get_matrix_values(), get_b_values())),
)

# ---- dolittle_fac Screen ----
dolittle_fac = tk.Frame(window, bg=main_bg)

tk.Label(
    dolittle_fac,
    text="Método de Factorizacion de Dolittle",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

df_button = tk.Button(
    dolittle_fac,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.dolittle_fac, (get_matrix_values(), get_b_values())
    ),
)

# ---- cholesky_factorization Screen ----
cholesky_factorization = tk.Frame(window, bg=main_bg)

tk.Label(
    cholesky_factorization,
    text="Método de Factorizacion de Cholesky",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

cf_button = tk.Button(
    cholesky_factorization,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.cholesky_factorization, (get_matrix_values(), get_b_values())
    ),
)

# ---- seidel Screen ----
seidel = tk.Frame(window, bg=main_bg)
seidel_ins = tk.Frame(seidel, bg=main_bg)
seidel_ins.pack(side="left", padx=30)

tk.Label(
    seidel_ins,
    text="Método de Seidel",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

tk.Label(
    seidel,
    bg=main_bg,
).pack(pady=17)


def se_on_select(event):
    global se_selected_option, se_dropdown
    se_selected_option.ser(se_dropdown.get())


se_labels = ["Tolerancia:", "Iteraciones:"]
se_entries = [None] * len(se_labels)
count = 0

for label_text in se_labels:
    tk.Label(
        seidel_ins,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    se_entries[count] = tk.Entry(seidel_ins, font=(font, size1))
    se_entries[count].pack()
    count += 1


tk.Label(
    seidel_ins,
    text="err type",
    font=(font, size),
    bg=main_bg,
    pady=10,
).pack()

se_selected_option = tk.StringVar()
se_selected_option.set("abs")
se_dropdown = tk.OptionMenu(
    seidel_ins,
    se_selected_option,
    *["abs", "rel"],
    command=se_on_select,
)
se_dropdown.config(
    font=(font, size),
    width=15,
)
se_dropdown.pack()


se_button = tk.Button(
    seidel,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.seidel,
        (
            get_matrix_values(),
            get_b_values(),
            float(se_entries[0].get()),
            float(se_entries[1].get()),
            se_selected_option,
        ),
    ),
)

# ---- jacobi Screen ----
jacobi = tk.Frame(window, bg=main_bg)
jacobi_ins = tk.Frame(jacobi, bg=main_bg)
jacobi_ins.pack(side="left", padx=30)

tk.Label(
    jacobi_ins,
    text="Método de Jacobi",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

tk.Label(
    jacobi,
    bg=main_bg,
).pack(pady=17)


def j_on_select(event):
    global j_selected_option, j_dropdown
    j_selected_option.ser(j_dropdown.get())


j_labels = ["Tolerancia:", "Iteraciones:"]
j_entries = [None] * len(j_labels)
count = 0

for label_text in j_labels:
    tk.Label(
        jacobi_ins,
        text=label_text,
        font=(font, size),
        bg=main_bg,
        pady=10,
    ).pack()

    j_entries[count] = tk.Entry(jacobi_ins, font=(font, size1))
    j_entries[count].pack()
    count += 1


tk.Label(
    jacobi_ins,
    text="err type",
    font=(font, size),
    bg=main_bg,
    pady=10,
).pack()

j_selected_option = tk.StringVar()
j_selected_option.set("abs")
j_dropdown = tk.OptionMenu(
    jacobi_ins,
    j_selected_option,
    *["abs", "rel"],
    command=j_on_select,
)
j_dropdown.config(
    font=(font, size),
    width=15,
)
j_dropdown.pack()


j_button = tk.Button(
    jacobi,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.jacobi,
        (
            get_matrix_values(),
            get_b_values(),
            float(j_entries[0].get()),
            float(j_entries[1].get()),
            j_selected_option,
        ),
    ),
)

# ---- vandermonde_method Screen ----
vandermonde_method = tk.Frame(window, bg=main_bg)

tk.Label(
    vandermonde_method,
    text="Método de Vandermonde",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)


v_button = tk.Button(
    vandermonde_method,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.vandermonde_method,
        (
            get_x_values(),
            get_y_values(),
        ),
    ),
)

# ---- newton_interpolacion Screen ----
newton_interpolacion = tk.Frame(window, bg=main_bg)

tk.Label(
    newton_interpolacion,
    text="Método de Interpolacion de Newton",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)


ni_button = tk.Button(
    newton_interpolacion,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.newton_interpolacion,
        (
            get_x_values(),
            get_y_values(),
        ),
    ),
)

# ---- spline Screen ----
spline = tk.Frame(window, bg=main_bg)

tk.Label(
    spline,
    text="Método de Splines",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)

tk.Label(
    spline,
    text="Valor de d:",
    font=(font, size),
    bg=main_bg,
).pack(pady=20)

s_entry = tk.Entry(spline, font=(font, size1))
s_entry.pack()

s_button = tk.Button(
    spline,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.spline,
        (
            get_x_values(),
            get_y_values(),
            float(s_entry.get()),
        ),
    ),
)

# ---- lagrange Screen ----
lagrange = tk.Frame(window, bg=main_bg)

tk.Label(
    lagrange,
    text="Método de Interpolacion de Newton",
    font=(font, size2, "bold"),
    bg=main_bg,
).pack(pady=20)


l_button = tk.Button(
    lagrange,
    text="Resolver",
    font=(font, size1),
    bg=button_bg,
    command=lambda: show_result(
        methods.lagrange,
        (
            get_x_values(),
            get_y_values(),
        ),
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
    ("Método de Gauss de pivote parcial", get_matrix, gauss_partial_pivot, gpp_button),
    ("Método de Gauss de pivote total", get_matrix, gauss_total_pivot, gtp_button),
    ("Método de LU de Gauss", get_matrix, lu_gauss, lug_button),
    (
        "Método de LU Descomposicion parcial",
        get_matrix,
        LU_partial_decomposition,
        lupd_button,
    ),
    ("Método de Crout", get_matrix, crout, c_button),
    ("Método de Factorizacion de Dolittle", get_matrix, dolittle_fac, df_button),
    (
        "Método de Factorizacion de Cholesky",
        get_matrix,
        cholesky_factorization,
        cf_button,
    ),
    ("Método de Seidel", get_matrix, seidel, se_button),
    ("Método de Jacobi", get_matrix, jacobi, j_button),
    ("Método de Vandermonde", create_points, vandermonde_method, v_button),
    ("Método de Interpolacion Newton", create_points, newton_interpolacion, ni_button),
    ("Método de Splines", create_points, spline, s_button),
    ("Método de Lagrange", create_points, lagrange, l_button),
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
    elif com == get_matrix:
        tk.Button(
            main_screen,
            text=text,
            font=(font, size1),
            bg=button_bg,
            width=button_width,
            command=lambda screen=screen, button=button: get_matrix(screen, button),
        ).grid(row=row, column=col, padx=5, pady=5)
    else:
        tk.Button(
            main_screen,
            text=text,
            font=(font, size1),
            bg=button_bg,
            width=button_width,
            command=lambda screen=screen, button=button: create_points(screen, button),
        ).grid(row=row, column=col, padx=5, pady=5)

    col += 1
    if col > 2:
        col = 0
        row += 1


def set_default_values(mat=0):
    global defaults, matrix1_entry
    entries_list = [
        is_entries,
        b_entries,
        rf_entries,
        pf_entries,
        nr_entries,
        s_entries,
        mr_entries,
        j_entries,
        se_entries,
        [matrix1_entry],
    ]
    if defaults:
        for entries in entries_list:
            for entry in entries:
                entry.delete(0, tk.END)

        values = ["x^3+4^x^2-10", "0", "0.1", "100"]
        for entry, value in zip(is_entries, values):
            entry.insert(0, value)
        values = ["x^3+4^x^2-10", "1", "2", "0.001", "100"]
        for entry, value in zip(b_entries, values):
            entry.insert(0, value)
        values = ["x^3+4^x^2-10", "1", "2", "0.001", "100", "1"]
        for entry, value in zip(rf_entries, values):
            entry.insert(0, value)
        values = [
            "log(sin(x)^2 + 1)-(1/2)-x",
            "log(sin(x)^2 + 1)-(1/2)",
            "-0.5",
            "0.0000001",
            "100",
        ]
        for entry, value in zip(pf_entries, values):
            entry.insert(0, value)
        values = [
            "log(sin(x)^2 + 1)-(1/2)",
            "2*(1/(sin(x)^2 + 1))*(sin(x)*cos(x))",
            "0.5",
            "0.0000001",
            "100",
        ]
        for entry, value in zip(nr_entries, values):
            entry.insert(0, value)
        values = ["log(sin(x)^2 + 1)-(1/2)", "0.5", "1", "0.0000001", "100"]
        for entry, value in zip(s_entries, values):
            entry.insert(0, value)
        values = ["x^3+4*x^2-10", "3*x^2+8*x", "6*x+8", "1", "0.0000001", "100"]
        for entry, value in zip(mr_entries, values):
            entry.insert(0, value)

        values = ["0.001", "100"]
        for entry, value in zip(j_entries, values):
            entry.insert(0, value)
        for entry, value in zip(se_entries, values):
            entry.insert(0, value)

        matrix1_entry.insert(0, "3")
        if mat == 3:
            values = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
            for row, value in zip(matrix_entries, values):
                for entry, val in zip(row, value):
                    entry.insert(0, val)
            values = [1, -2, 5]
            for entry, value in zip(matb_entries, values):
                entry.insert(0, value)
        if mat == 4:
            values = [
                [4, -1, 0, 3],
                [1, 15.5, 3, 8],
                [0, -1.3, 4, 1.1],
                [14, 5, -2, 30],
            ]
            for row, value in zip(matrix_entries, values):
                for entry, val in zip(row, value):
                    entry.insert(0, val)
            values = [1, 1, 1, 1]
            for entry, value in zip(matb_entries, values):
                entry.insert(0, value)

        add_row()
        add_row()
        add_row()
        values = ["-1", "0", "3", "4"]
        for entry, value in zip(pointsx_entries, values):
            entry.insert(0, value)
        values = ["15.5", "3", "8", "1"]
        for entry, value in zip(pointsy_entries, values):
            entry.insert(0, value)

    else:
        for entries in entries_list:
            for entry in entries:
                entry.delete(0, tk.END)


# Initially show the input screen
main_screen.pack()
# Run the application
window.mainloop()
