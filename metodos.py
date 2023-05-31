import pandas as pd
import numpy as np
import math
import sympy as sym
from scipy import linalg

def incremental_search(f, x_init, delta_x, num_iterations):
    intervals = []
    message = []
    x = x_init
    fx = fun(x, f)
    for i in range(1, num_iterations):
        x_new = x + delta_x
        fx_new = fun(x_new, f)
        if fx * fx_new <= 0:
            intervals.append((i, x, x_new))
        x = x_new
        fx = fx_new
        if fx == 0:
            intervals.append((i, x_new, x_new))
            x = x_new
            fx = fun(x, f)
    if intervals:
        util = "Se encontraron intervalos en los que f(x) cambia de signo"
    else:
        util = "No se encontraron intervalos en los que f(x) cambia de signo en el número máximo de iteraciones"
    for i, interval in enumerate(intervals):
        message.append(f"Intervalo {i+1}: {interval}")
    print(util)
    message.append(util)
    return message

def fun(x, expression):
    result = sym.simplify(expression)
    substituted = result.subs("x", x)
    return substituted

def biseccion(f, xi, xs, tol, niter):
    Xi = xi
    Xs = xs
    Tol = tol
    Niter = niter
    Fun = f
    res = []
    fm = []
    E = []
    xn = []
    N = []
    a = []
    b = []
    c = 0
    x = Xi
    fi = eval(Fun)
    x = Xs
    fs = eval(Fun)
    Error = 0
    res2 = "nan"
    if fi == 0:
        s = Xi
        E = 0
        res2 = (Xi, "es raiz de f(x)")
    elif fs == 0:
        s = Xs
        E = 0
        res2 = (Xs, "es raiz de f(x)")
    elif fs * fi < 0:
        c = 0
        Xm = (Xi + Xs) / 2
        x = Xm
        fe = eval(Fun)
        fm.append(fe)
        N.append(c)
        E.append(100)
        xn.append(x)
        while E[c] > Tol and fe != 0 and c < Niter:
            a.append(Xi)
            b.append(Xs)
            if fi * fe < 0:
                Xs = Xm
                x = Xs
                fs = eval(Fun)
            else:
                Xi = Xm
                x = Xi
                fs = eval(Fun)
            Xa = Xm
            Xm = (Xi + Xs) / 2
            x = Xm
            fe = eval(Fun)
            fm.append(fe)
            xn.append(x)
            Error = abs(Xm - Xa)
            E.append(Error)
            c = c + 1
            N.append(c)
        a.append(Xi)
        b.append(Xs)
        if fe == 0:
            s = x
            res2 = (s, "es raiz de f(x)")
        elif Error < Tol:
            s = x
            res2 = (s, "es una aproximacion de un raiz de f(x) con una tolerancia", Tol)
        else:
            s = x
            res2 = ("Fracaso en ", Niter, " iteraciones")
    else:
        res2 = "El intervalo es inadecuado"

    for i in range(0, len(N)):
        res.append([N[i], a[i], xn[i], b[i], fm[i], E[i]])
    return res, res2

def regla_falsa(f, x_lower, x_upper, tol, max_iterations, t_error):
    results = []
    fx_lower = fun_false(x_lower, f)
    fx_upper = fun_false(x_upper, f)
    if fx_lower == 0:
        return [], f"{x_lower} es raíz"
    elif fx_upper == 0:
        return [], f"{x_upper} es raíz"
    elif fx_lower * fx_upper < 0:
        x = (x_lower * fx_upper - x_upper * fx_lower) / (fx_upper - fx_lower)
        fx = fun_false(x, f)
        iteration = 1
        error = tol + 1
        results.append([iteration, x_lower, x_upper, x, fx, float("nan")])
        while fx != 0 and error > tol and iteration < max_iterations:
            if fx_lower * fx < 0:
                x_upper = x
                fx_upper = fx
            else:
                x_lower = x
                fx_lower = fx
            x_prev = x
            x = (x_lower * fx_upper - x_upper * fx_lower) / (fx_upper - fx_lower)
            fx = fun_false(x, f)
            if t_error == 1:
                error = abs((x - x_prev) / x)
            else:
                error = abs(x - x_prev)
            iteration += 1
            results.append([iteration, x_lower, x_upper, x, fx, error])
        if fx == 0:
            return results, f"{x} es raíz"

        elif error <= tol:
            return results, f"{x} se aproxima a una raíz con una tolerancia de {tol}"

        else:
            return results, "Se alcanzó el número máximo de iteraciones"
    else:
        return [], "El intervalo es inadecuado"

def fun_false(x, expression):
    result = eval(expression)
    return result

def fixed_point(f, g, x0, tol, max_iterations):
    results = []
    x = x0
    fx = fun_fijo(x, f)
    iteration = 0
    error = tol + 1
    results.append([iteration, x, fx, g_fun_fijo(x, g), float("nan")])
    while fx != 0 and error > tol and iteration < max_iterations:
        x = g_fun_fijo(x, g)
        fx = fun_fijo(x, f)
        error = abs(results[-1][1] - x)
        iteration += 1
        results.append([iteration, x, fx, g_fun_fijo(x, g), error])
    if fx == 0:
        return results, f"{x} es raíz"
    elif error <= tol:
        return results, f"{x} se aproxima a una raíz con una tolerancia de {tol}"
    else:
        return results, "Se alcanzó el número máximo de iteraciones"

def fun_fijo(x, expression):
    result = sym.simplify(expression)
    substituted = result.subs("x", x)
    return substituted

def g_fun_fijo(x, expression):
    result = sym.simplify(expression)
    substituted = result.subs("x", x)
    return substituted

def newton_raphson(f, df, x0, tol, max_iterations):
    results = []
    x = x0
    fx = eval_f(f, x)
    dfx = eval_df(df, x)
    iteration = 0
    error = tol + 1
    results.append(
        [
            iteration,
            "{:.10f}".format(x),
            "{:.1e}".format(fx).replace("e-0", "e-"),
            "NaN",
        ]
    )
    while fx != 0 and dfx != 0 and error > tol and iteration < max_iterations:
        x_prev = x
        x -= fx / dfx
        fx = eval_f(f, x)
        dfx = eval_df(df, x)
        error = abs(x_prev - x)
        iteration += 1
        results.append(
            [
                iteration,
                "{:.10f}".format(x),
                "{:.1e}".format(fx).replace("e-0", "e-"),
                "{:.1e}".format(error).replace("e-0", "e-"),
            ]
        )
    if fx == 0:
        return (results,)
    elif error <= tol:
        util = (
            "An approximation of the roof was found for x"
            + str(len(results) - 1)
            + " "
            + str(results[-1][1])
        )
        return results, util
    else:
        return (
            results,
            "Given the number of iterations and the tolerance, it was impossible to find a satisfying root",
        )

def eval_f(f, x):
    result = sym.simplify(f)
    substituted = result.subs("x", x)
    return substituted

def eval_df(f, x):
    result = sym.simplify(f)
    substituted = result.subs("x", x)
    return substituted

def secante(f, x0, x1, tol, max_iterations):
    results = []
    fx0 = fun_secan(x0,f)
    fx1 = fun_secan(x1,f)
    iteration = 0
    error = tol + 1
    results.append([iteration, x0, fx0, float('nan')])
    iteration += 1
    results.append([iteration, x1, fx1, abs(x1 - x0)])
    
    while fx1 != 0 and error > tol and iteration < max_iterations:
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = fun_secan(x2,f)
        error = abs(x2 - x1)
        iteration += 1
        results.append([iteration, x2, fx2, error])
        
        x0 = x1
        fx0 = fx1
        x1 = x2
        fx1 = fx2

    if fx1 == 0:
        return results, f"{x1} es raíz"
    elif error <= tol:
        return results, f"{x1} se aproxima a una raíz con una tolerancia de {tol}"
    else:
        return results, "Se alcanzó el número máximo de iteraciones"

def fun_secan(x, expression):
    result = sym.simplify(expression)
    substituted = result.subs('x', x)
    return substituted

def multiple_roots(f, df, df2, x0, tol, max_iterations):
    results = []
    valorX = {'x': x0}
    fx = eval(f, globals(), valorX)
    dfx = eval(df, globals(), valorX)
    df2x = eval(df2, globals(), valorX)
    iteration = 0
    error = tol + 1
    results.append([iteration, '{:.10f}'.format(x0), '{:.1e}'.format(fx).replace('e-0', 'e-'), 'NaN'])
    while fx != 0 and dfx != 0 and error > tol and iteration < max_iterations:
        numerator = fx * dfx
        denominator = dfx**2 - fx * df2x
        x1 = x0 - numerator / denominator
        valorX['x'] = x1
        fx = eval(f, globals(), valorX)
        dfx = eval(df, globals(), valorX)
        df2x = eval(df2, globals(), valorX)
        error = abs(x1 - x0)
        iteration += 1
        results.append([iteration, '{:.10f}'.format(x1), '{:.1e}'.format(fx).replace('e-0', 'e-'),'{:.1e}'.format(error).replace('e-0', 'e-')])
        x0 = x1
    if fx == 0:
        return results, f"An approximation of the roof was found for m = {x0}"
    elif error <= tol:
        return results, f"An approximation of the roof was found for m = {x0}"
    else:
        return results, "Given the number of iterations and the tolerance, it was impossible to find a satisfying root"

def simple_gauss(a, b):
    ab = to_aug(a, b)
    res = []
    res.append(np.copy(ab).tolist())

    size = len(a)

    # Stages

    for i in range(0, size - 1):
        # Compute multiplier for row in stage.
        for j in range(i + 1, size):
            multiplier = ab[j][i] / ab[i][i]
            for k in range(i, size + 1):
                ab[j][k] = ab[j][k] - (multiplier * ab[i][k])
        res.append(np.copy(ab).tolist())

    return res

def to_aug(a, b):
    return np.column_stack((a, b))

def regressive_substitution(ab, labels=None):
    size = ab.shape[0]
    assert ab.shape[1] == size + 1

    solutions = np.zeros(size, dtype=np.float64)
    solutions[size - 1] = ab[size - 1][size] / ab[size - 1][size - 1]

    # Loop backwards
    for i in range(size - 2, -1, -1):
        accum = 0
        for p in range(i + 1, size):
            accum += ab[i][p] * solutions[p]
        solutions[i] = (ab[i][size] - accum) / ab[i][i]

    # Update the labels and assign its values
    labeled_xs = np.zeros(size)
    if labels is not None:
        for i, v in enumerate(labels):
            labeled_xs[labels[i]] = solutions[i]
        solutions = labeled_xs

    return solutions

def progressive_substitution(ab):
    size = ab.shape[0]
    assert ab.shape[1] == size + 1

    solutions = np.zeros(size, dtype=np.float64)
    solutions[0] = ab[0][size] / ab[0][0]

    for i in range(1, size):
        accum = 0
        for p in range(0, i):
            accum += ab[i][p] * solutions[p]

        solutions[i] = (ab[i][size] - accum) / ab[i][i]
    return solutions

def gauss_partial_pivot(a, b):
    ab = to_aug(a, b)
    res = []
    res.append(np.copy(ab).tolist())
    size = len(a)
    # Stages
    for k in range(0, size - 1):
        partial_pivot(ab, k)
        # Compute multiplier for row in stage.
        for i in range(k + 1, size):
            multiplier = ab[i][k] / ab[k][k]
            for j in range(k, size + 1):
                ab[i][j] = ab[i][j] - (multiplier * ab[k][j])
        res.append(np.copy(ab).tolist())
    return res

def gauss_total_pivot(a, b):
    ab = to_aug(a, b)
    res = []
    res.append(np.copy(ab).tolist())
    size = len(a)
    labels = list(range(0, size))
    # Stages
    for k in range(0, size - 1):
        total_pivot(ab, k, labels)
        # Compute multiplier for row in stage.
        for i in range(k + 1, size):
            multiplier = ab[i][k] / ab[k][k]
            for j in range(k, size + 1):
                ab[i][j] = ab[i][j] - (multiplier * ab[k][j])
        res.append(np.copy(ab).tolist())
    return res, labels

def partial_pivot(ab, k):
    largest = abs(ab[k][k])
    largest_row = k
    size = ab.shape[0]

    for r in range(k + 1, size):
        current = abs(ab[r][k])
        if current > largest:
            largest = current
            largest_row = r
    if largest == 0:
        raise Exception("Equation system does not have unique solution.")
    else:
        if largest_row != k:
            ab[[k, largest_row]] = ab[[largest_row, k]]

def total_pivot(ab, k, labels):
    largest = abs(ab[k][k])
    largest_row = k
    largest_column = k

    size = ab.shape[0]
    # i itera filas, j columnas
    for i in range(k, size):
        for j in range(k, size):
            current = abs(ab[i][j])
            if current > largest:
                largest = current
                largest_row = i
                largest_column = j
    if largest == 0:
        raise Exception("Equation system does not have unique solution.")
    else:
        if largest_row != k:
            ab[[k, largest_row]] = ab[[largest_row, k]]
        if largest_column != k:
            ab[:, [k, largest_column]] = ab[:, [largest_column, k]]
            labels[k], labels[largest_column] = labels[largest_column], labels[k]

def lu_gauss(a, b):
    res = []
    size = len(a)
    # U
    # L
    lower_tri = np.identity(size, dtype=np.float64)
    # a = M
    # Stages
    for k in range(0, size - 1):
        # Compute multiplier for row in stage.
        for i in range(k + 1, size):
            multiplier = a[i][k] / a[k][k]
            for j in range(k, size):
                a[i][j] = a[i][j] - (multiplier * a[k][j])
                if i > j:
                    lower_tri[i][j] = multiplier
        u = np.copy(a)
        aux = k
        for i in range(aux + 2, size):
            u[i] = 0
        res.append([u.tolist(), np.copy(lower_tri).tolist(), np.copy(a).tolist()])
    z = progressive_substitution(to_aug(lower_tri, b))
    return res, regressive_substitution(to_aug(a, z))

def LU_partial_decomposition(A, B):
    n, m = A.shape
    P = np.identity(n)
    L = np.identity(n)
    U = A.copy()
    PF = np.identity(n)
    LF = np.zeros((n, n))
    for k in range(0, n - 1):
        index = np.argmax(abs(U[k:, k]))
        index = index + k
        if index != k:
            P = np.identity(n)
            P[[index, k], k:n] = P[[k, index], k:n]
            U[[index, k], k:n] = U[[k, index], k:n]
            PF = np.dot(P, PF)
            LF = np.dot(P, LF)
        L = np.identity(n)
        for j in range(k + 1, n):
            L[j, k] = -(U[j, k] / U[k, k])
            LF[j, k] = U[j, k] / U[k, k]
        U = np.dot(L, U)
    np.fill_diagonal(LF, 1)
    for i in range(L.shape[0]):
        for j in range(i):
            B[i] -= L[i, j] * B[j]
    # Sustitución regresiva
    for i in range(U.shape[0] - 1, -1, -1):
        for j in range(i + 1, U.shape[1]):
            B[i] -= U[i, j] * B[j]
        B[i] = B[i] / U[i, i]
    return PF, LF, U, B

def crout(a, b):
    n = a.shape[0]
    lower_tri = np.identity(n, dtype=np.float64)
    upper_tri = np.identity(n, dtype=np.float64)
    res = []
    for k in range(0, n):
        sum0 = 0

        for p in range(0, k):
            sum0 += lower_tri[k][p] * upper_tri[p][k]
        lower_tri[k][k] = a[k][k] - sum0

        for i in range(k + 1, n):
            sum1 = 0
            for p in range(0, k):
                sum1 += lower_tri[i][p] * upper_tri[p][k]
            lower_tri[i][k] = a[i][k] - sum1

        for j in range(k + 1, n):
            sum2 = 0
            for p in range(0, k):
                sum2 += lower_tri[k][p] * upper_tri[p][j]
            upper_tri[k][j] = (a[k][j] - sum2) / lower_tri[k][k]
        res.append([np.copy(upper_tri), np.copy(lower_tri)])
    z = progressive_substitution(to_aug(lower_tri, b))
    return res, regressive_substitution(to_aug(upper_tri, z))

def dolittle_fac(a, b):
    size = a.shape[0]
    lower_tri = np.identity(size, dtype=np.float64)
    upper_tri = np.identity(size, dtype=np.float64)
    res = []
    for k in range(0, size):
        first_sum = 0
        for p in range(0, k):
            first_sum += lower_tri[k][p] * upper_tri[p][k]
        upper_tri[k][k] = a[k][k] - first_sum
        for i in range(k + 1, size):
            second_sum = 0
            for p in range(0, k):
                second_sum += lower_tri[i][p] * upper_tri[p][k]
            lower_tri[i][k] = (a[i][k] - second_sum) / upper_tri[k][k]
        for j in range(k + 1, size):
            third_sum = 0
            for p in range(0, k):
                third_sum += lower_tri[k][p] * upper_tri[p][j]
            upper_tri[k][j] = a[k][j] - third_sum
        res.append([np.copy(upper_tri), np.copy(lower_tri)])

    z = progressive_substitution(to_aug(lower_tri, b))
    return res, regressive_substitution(to_aug(upper_tri, z))

def cholesky_factorization(a, b):
    size = a.shape[0]
    lower_tri = np.identity(size, dtype=np.float64)
    upper_tri = np.identity(size, dtype=np.float64)
    res = []
    for k in range(0, size):
        first_sum = 0
        # Compute lower_tri[k][k]
        for p in range(0, k):
            first_sum += lower_tri[k][p] * upper_tri[p][k]
        upper_tri[k][k] = np.sqrt(a[k][k] - first_sum)
        lower_tri[k][k] = upper_tri[k][k]

        # Compute lower_tri[i][k]
        for i in range(k + 1, size):
            second_sum = 0
            for p in range(0, k):
                second_sum += lower_tri[i][p] * upper_tri[p][k]
            lower_tri[i][k] = (a[i][k] - second_sum) / lower_tri[k][k]

        # Compute upper_tri[k][j]
        for j in range(k + 1, size):
            third_sum = 0
            for p in range(0, k):
                third_sum += lower_tri[k][p] * upper_tri[p][j]
            upper_tri[k][j] = (a[k][j] - third_sum) / upper_tri[k][k]
        res.append([np.copy(upper_tri), np.copy(lower_tri)])
    z = progressive_substitution(to_aug(lower_tri, b))
    return res, regressive_substitution(to_aug(upper_tri, z))

def seidel(a, b, init, tol, n, err_type="abs"):
    table = []
    res = []
    error = float("inf")
    xn = init
    i = 0
    table.append(i)
    table.append(xn)
    table.append("")
    table.append("")
    table.append("newline")
    res.append([i, xn.tolist(), "nan"])
    while error > tol and i < n:
        x, abs_err, rel_err = next_iter(a, b, xn)
        xn = x
        if err_type == "rel":
            error = rel_err
        else:
            error = abs_err
        i += 1
        res.append([i, xn.tolist(), abs_err])
    return xn, res

def next_iter(a, b, prev_x):
    size = a.shape[0]
    x = np.copy(prev_x)

    for i in range(0, size):
        d = a[i][i]
        accum = 0
        for j in range(0, size):
            if j != i:
                accum += a[i][j] * x[j]
        x[i] = (b[i] - accum) / d

    errs = abs(x - prev_x)
    abs_err = max(errs)
    rel_err = max(errs / abs(x))

    return x, abs_err, rel_err

def jacobi(a, b, init, tol, n, err_type="abs"):
    table = []
    assert a.shape[0] == a.shape[1]
    assert a.shape[0] == len(b)
    assert len(init) == len(b)
    res = []
    error = float("inf")

    xn = init
    i = 0

    table.append(i)
    table.append(xn)
    table.append("")
    table.append("")
    table.append("newline")
    res.append([i, xn.tolist(), "nan"])
    while error > tol and i < n:
        x, abs_err, rel_err = next_iter2(a, b, xn)
        xn = x

        if err_type == "rel":
            error = rel_err
        else:
            error = abs_err

        i += 1

        res.append([i, xn.tolist(), abs_err])
        table.append("newline")
    return xn, res

def next_iter2(a, b, prev_x):
    size = a.shape[0]
    x = np.zeros(size, dtype=np.float64)

    for i in range(0, size):
        d = a[i][i]
        accum = 0
        for j in range(0, size):
            if j != i:
                accum += a[i][j] * prev_x[j]
        x[i] = (b[i] - accum) / d

    errs = abs(x - prev_x)
    abs_err = max(errs)
    rel_err = max(errs / abs(x))

    return x, abs_err, rel_err

def sor(A, x0, b, Tol, niter, w):
    c = 0
    E = []
    resultado = []
    error = Tol + 1
    E.append(error)
    D = np.diagonal(A) * np.identity(len(x0))
    L = -np.tril(A, -1)
    U = -np.triu(A, +1)
    resultado.append([c, x0, error])
    while error > Tol and c < niter:
        T = np.dot(np.linalg.inv(D - (w * L)), ((1 - w) * D + (w * U)))
        C = w * np.dot(np.linalg.inv(D - w * L), b)
        x1 = np.dot(T, x0) + C
        E.append(np.linalg.norm(x1 - x0))
        error = E[c]
        x0 = x1
        c = c + 1
        resultado.append([c, x0.tolist(), error])
        if error < Tol:
            s = x0
            n = c
            return [
                resultado,
                "Solucion al sistema con una tolerancia de "
                + str(Tol)
                + " es "
                + str(s),
            ]
    s = x0
    n = c
    return [resultado, "Fracasó" + str(niter)]

def vandermonde_method(x, y):
    matrix = []
    coeficientes = []
    xn = np.array(x)
    yn = np.array([y]).T
    A = np.vander(xn)
    Ainv = np.linalg.inv(A)
    a = np.dot(Ainv, yn)
    matrix = A
    coeficientes = a
    return {"matrix": (matrix), "coeficients": (coeficientes)}

def newton_interpolacion(x, y):
    n = len(y)
    Tabla = np.zeros([n, n])
    Tabla[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            Tabla[i][j] = (Tabla[i + 1][j - 1] - Tabla[i][j - 1]) / (x[i + j] - x[i])
    return {"table": (Tabla).tolist(), "coef": (Tabla[0]).tolist()}

def spline(x, y):
    x = np.array(x)
    y = np.array(y)
    n = x.size
    m = 4*(n-1)
    A = np.zeros((m,m))
    b = np.zeros((m,1))
    Coef = np.zeros((n-1,4))
    i = 0
    #Interpolating condition
    while i < x.size-1:
        A[i+1,4*i:4*i+4]= np.hstack((x[i+1]**3,x[i+1]**2,x[i+1],1)) 
        b[i+1]=y[i+1]
        i = i+1
    A[0,0:4] = np.hstack((x[0]**3,x[0]**2,x[0]**1,1))
    b[0] = y[0]
    #Condition of continuity
    i = 1
    while i < x.size-1:
        A[x.size-1+i,4*i-4:4*i+4] = np.hstack((x[i]**3,x[i]**2,x[i],1,-x[i]**3,-x[i]**2,-x[i],-1))
        b[x.size-1+i] = 0
        i = i+1
    #Condition of smoothness
    i = 1
    while i < x.size-1:
        A[2*n-3+i,4*i-4:4*i+4] = np.hstack((3*x[i]**2,2*x[i],1,0,-3*x[i]**2,-2*x[i],-1,0))
        b[2*n-3+i] = 0
        i = i+1
    #Concavity condition
    i = 1
    while i < x.size-1:
        A[3*n-5+i,4*i-4:4*i+4] = np.hstack((6*x[i],2,0,0,-6*x[i],-2,0,0))
        b[n+5+i] = 0
        i = i+1
    #Boundary conditions  
    A[m-2,0:2]=[6*x[0],2]
    b[m-2]=0
    A[m-1,m-4:m-2]=[6*x[x.size-1],2]
    b[m-1]=0
    Saux = linalg.solve(A,b)
    #Order Coefficients
    i = 0
    j = 0
    while i < n-1:
        Coef[i,:] = np.hstack((Saux[j],Saux[j+1],Saux[j+2],Saux[j+3]))
        i = i+1
        j = j + 4
    output = Coef
    return output
def lagrange(arreglo_x, arreglo_y):
    puntos=[]
    puntos.append(arreglo_x)
    puntos.append(arreglo_y)
    x = sym.Symbol("x")
    size = np.size(puntos, 0)
    producto = 0
    ls = []
    for k in range(size):
        l = 1
        for i in range(size):
            if i != k:
                l = l * ((x - arreglo_x[i]) / (arreglo_x[k] - arreglo_x[i]))
        ls.append(l)
        producto = producto + l * (arreglo_y[k])
    producto = sym.simplify(sym.expand(producto))
    return producto, ls


vari1, vari2=seidel([[4,-1,0,3], [1,15.5,3,8], [0,-1.3,4,1.1],[14,5,-2,30]], [1,1,1,1],[0,0,0,0],0.0000001,100)
print("Hola")
#biseccion("x**3+4**x**2-10", 1, 2, 0.001, 100)
#regla_falsa("x**3+4**x**2-10", 1, 2, 0.001, 100)
