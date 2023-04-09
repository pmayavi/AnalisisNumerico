#f funcion continua
#a extremo derecho del intervalo inicial
#b extremo final del intervalo final
#tol tolerancia
#Nmax numero maximo de iteraciones
#iter numero de iteraciones
#err error
def biseccion(f, a, b, tol, Nmax):
    fa = f(a)
    pm = (a+b)/2
    fpm = f(pm)
    E = 1000
    cont = 1
    while E > tol and cont < Nmax:
        if fa*fpm < 0:
            b = pm
        else:
            a = pm
            fa = fpm
        p0 = pm
        pm = (a+b)/2
        fpm = f(pm)
        E = abs(pm-p0)
        cont += 1
    x = pm
    iter = cont
    err = E
    return x, iter, err

#f, funcion continua
#a, extremo derecho del intervalo inicial
#b, extremo final del intervalo final
#tol, tolerancia
#Nmax, maximo de iteraciones
def reglafalsa(f, a, b, tol, Nmax):
    fa = f(a)
    fb = f(b)
    pm = (fb * a - fa * b) / (fb - fa)
    fpm = f(pm)
    E = 1000
    cont = 1
    
    while E > tol and cont < Nmax:
        if fa * fpm < 0:
            b = pm
        else:
            a = pm
        p0 = pm
        pm = (f(b) * a - f(a) * b) / (f(b) - f(a))
        fpm = f(pm)
        E = abs(pm - p0)
        cont = cont + 1
        
    x = pm
    iter = cont
    err = E
    
    return x, iter, err

#f función continua
#g función continua
#x0 aproximación inicial
#tol tolerancia
#Nmax número máximo de iteraciones
#x solución
#iter número de iteraciones
#err error
def punto_fijo(f, g, x0, tol, Nmax):
    xant = x0
    E = 1000
    cont = 0
    while E > tol and cont < Nmax:
        xact = g(xant)
        E = abs(xact - xant)
        cont += 1
        xant = xact
    x = xact
    iter = cont
    err = E
    return x, iter, err

#f, funcion continua
#f', funcion continua
#x0, aproximacion inicial 
#tol, tolerancia
#Nmax, maximo de iteraciones
def newton(f, df, x0, tol, Nmax):
    xant = x0
    fant = f(xant)
    E = 1000
    cont = 0
    
    while E > tol and cont < Nmax:
        xact = xant - fant / df(xant)
        fact = f(xact)
        E = abs(xact - xant)
        cont = cont + 1
        xant = xact
        fant = fact
        
    x = xact
    iter = cont
    err = E
    
    return x, iter, err

#f función continua
#x0 aproximación inicial
#x1 aproximación inicial
#tol tolerancia
#Nmax: número máximo de iteraciones
#x solución
#iter número de iteraciones
#err error
def secante(f, x0, x1, tol, Nmax):
    # Inicialización
    f0 = f(x0)
    f1 = f(x1)
    E = 1000
    cont = 1
    while E > tol and cont < Nmax:
        xact = x1 - f1*(x1 - x0)/(f1 - f0)
        fact = f(xact)
        E = abs(xact - x1)
        cont += 1
        x0 = x1
        f0 = f1
        x1 = xact
        f1 = fact
    x = xact
    iter = cont
    err = E

    return x, iter, err

#Entradas: 
#f, funcion continua
#f', funcion continua
#f'', funcion continua
#x0, aproximacion inicial 
#tol, tolerancia
#Nmax, maximo de iteraciones
def raicesmlt(f, df, d2f, x0, tol, Nmax):
    xant = x0
    fant = f(xant)
    E = 1000 
    cont = 0
    
    while E > tol and cont < Nmax:
        xact = xant - fant * df(xant) / ((df(xant)) ** 2 - fant * d2f(xant))
        fact = f(xact)
        E = abs(xact - xant)
        cont = cont + 1
        xant = xact
        fant = fact
        
    x = xact
    iter = cont
    err = E
    
    return x, iter, err
