import numpy
import math
import Auxiliary_module

###############
# ACTIVIDAD 1 #
###############

def exercise_1():
    """
    ENUNCIADO:

    Calcular las componentes cartesianas del vector de deformación en cada
    estación. Calcular también el módulo de la deformación, el ángulo de
    elevación, el módulo de la deformación horizontal, y su azimut con respecto
    al norte.
    """

    #>>>>>
    # Importación de módulos y paquetes.
    #>>>>>
    """
    Sin los módulos apropiados, no se puede hacer nada. Para este ejercicio,
    se ha tenido que recurrir al paquete 'NumPy' para un manejo eficiente de
    secuencias ordenadas de números (arrays).
    """
    import numpy


    #>>>>>
    # Lectura de los datos.
    #>>>>>
    """
    Los datos están contenidos en dos archivos, 'data_file_1.txt' y
    'data_file_2.txt', y contienen las posiciones de los puntos en las
    mediciones primera y segunda, respectivamente.

    Dada la estructura con que están organizados los puntos, 'NumPy' ofrece una
    función que permite una fácil extracción de la información de cada archivo.
    """
    data_array_1 = numpy.loadtxt("data_file_1.txt", comments = "#")
    data_array_2 = numpy.loadtxt("data_file_2.txt", comments = "#")


    #>>>>>
    # Cálculo de los vectores de deformación, y guardado de los resultados.
    #>>>>>
    """
    'NumPy' ofrece un procedimiento de resta de matrices asombrosamente simple;
    tan simple como si se restasen dos números.
    """
    deformation_array = data_array_2 - data_array_1

    """
    Tras la operación, guardo el array resultante por si me pudiese hacer falta
    más adelante. Para ello, uso la función 'numpy.savetxt'.

    En el archivo he incluido una primera línea con un comentario aclaratorio, y
    he indicado que quiero que los números sean reales, con un máximo de tres
    dígitos como parte decimal y, al menos, siete caracteres para cada número.
    """
    numpy.savetxt("deformation_vectors.txt", deformation_array)


    #>>>>>
    # Cálculo de valores relacionados con los vectores.
    #>>>>>
    """
    La obtención de las cuatro cosas que se piden no es tan sencilla como lo de
    antes, de modo que toca calcular explícitamente cada valor.

    No obstante, primero voy a importar el módulo 'math'...
    """
    import math

    """
    ... y a crear un array de ceros con cuatro columnas y con un número de filas
    idéntico al de 'deformation_array'.
    """
    deformation_data = numpy.zeros((len(deformation_array), 4))

    """
    Para cada fila de 'deformation_array'...
    """
    for i in range(len(deformation_array)):

        """
        (PREVIO) Se calcula el módulo del vector en dos dimensiones descrito por
        las coordenadas X e Y.
        """

        horizontal_length = \
            math.sqrt(math.pow(deformation_array[i, 0], 2) + \
                      math.pow(deformation_array[i, 1], 2))

        """
        (1) Se calcula el módulo del vector de deformación, y se coloca su valor
            en la primera columna.
        """
        deformation_data[i, 0] = \
            math.sqrt(math.pow(horizontal_length, 2) + \
                      math.pow(deformation_array[i, 2], 2))

        """
        (2) Se calcula el ángulo de elevación con respecto a la horizontal, y se
            expresa en grados sexagesimales.
        """
        deformation_data[i, 1] = \
            (180.0 / math.pi) * \
            math.atan(deformation_array[i, 2] / horizontal_length)

        """
        (3) El valor calculado en el paso (PREVIO) resulta ser el elemento de la
            tercera columna.
        """
        deformation_data[i, 2] = horizontal_length

        """
        (4) Se obtiene el azimut del vector de deformación proyectado sobre el
            plano XY y expresado en grados sexagesimales. Como referencia, 0º
            indicará deformación hacia el norte; 90º, hacia el este.

        (4.1) El programa comenzará calculando el valor absoluto de este
              arcocoseno.
        """
        initial_guess = \
            abs(math.acos(deformation_array[i, 0] / horizontal_length))

        """
        (4.2) Si el arcoseno de 'deformation_array[i, 0] / horizontal_length' es
        negativo, 'initial_guess' seguirá tal cual. En caso contrario,
        initial_guess = 360.0 - initial_guess.
        """
        initial_guess = initial_guess if \
            math.asin(deformation_array[i, 1] / horizontal_length) < 0.0 \
            else (360.0 - initial_guess)

        """
        El proceso de conversión se completa sumando 90º a 'initial_guess'.
        Luego se coloca este valor en la cuarta columna de 'deformation_data'.
        """
        initial_guess += 90.0
        if (initial_guess >= 360.0):
            initial_guess -= 360.0
        deformation_data[i, 3] = initial_guess


    #>>>>>
    # Volcado de los datos en un archivo de texto para su posterior utilización.
    #>>>>>
    """
    Así evito arrastrar todas las referencias a los objetos y no me tengo que
    preocupar de tener cuidado con los nombres de las variables.
    """
    numpy.savetxt("deformation_data.txt", deformation_data, header = \
                  "Length (m) -- Pitch (º) (> 0º = Up)" + \
                  " -- Horizontal length (m)" + \
                  " -- Heading (º) (0º = North; 90º = East)")





###############
# ACTIVIDAD 2 #
###############

def exercise_2():
    """
    Representar gráficamente la deformación horizontal como vectores centrados
    en cada estación, cuya dirección coincide con la dirección de la deformación
    horizontal y cuyo módulo es proporcional al módulo de la deformación
    horizontal. Incluir también una elipse de error como estimación de la
    incertidumbre de la medida.
    """


    #>>>>>
    # Importación de módulos y paquetes.
    #>>>>>
    """
    Aquí, además del paquete 'NumPy', se necesita el paquete 'Matplotlib' (en
    concreto, el módulo 'pyplot') para realizar las representaciones gráficas.
    """
    import numpy
    from matplotlib import pyplot
    import matplotlib
    matplotlib.use("TkAgg")

    #>>>>>
    # Lectura de los archivos.
    #>>>>>
    """
    Para representar los cambios de cada punto, el programa recurrirá al archivo
    'deformation_vectors.txt'.
    """

    positions = numpy.loadtxt("data_file_1.txt", comments = "#", \
        usecols = (0, 1,))
    vectors = numpy.loadtxt("deformation_vectors.txt", comments = "#", \
        usecols = (0, 1,))
    ends_of_vectors = positions + (vectors * 100000.0)


    #>>>>>
    # Representación gráfica de las deformaciones en horizontal.
    #>>>>>
    """
    La idea es conocer cómo han variado las posiciones de los puntos de una
    medición a la siguiente.

    Para representar gráficas 2D con barras de errores, se puede utilizar la
    funcion 'pyplot.errorbar', que permite implementar barras de error con
    tamaños personalizados para los puntos que se desee.

    Dado que el margen de error es de 1mm para todos los vectores y en
    cualquiera de las direcciones de los ejes cartesianos, sólo necesito
    proporcionar el valor '0.001' a los parámetros 'yerr' y 'xerr'... Aunque se
    han reescalado los vectores y los errores para que sean visibles.

    'ecolor' define el color de las barras de error en caso de querer un color
    diferente del color de los marcadores de los puntos. En este caso, se ha
    optado por barras azules.

    Las elipses de error se han pintado de forma manual sobre la imagen una vez
    obtenida.
    """

    fig1 = pyplot.figure(1, dpi = 200, figsize = (10, 10))

    pyplot.plot(positions[:, 0], positions[:, 1], ".r")

    for i in range(len(positions)):
        pyplot.errorbar([positions[i, 0], ends_of_vectors[i, 0]], \
            [positions[i, 1], ends_of_vectors[i, 1]], fmt = "-b", \
            xerr = [0.0, 500.0], yerr = [0.0, 500.0], \
            ecolor = "#ff00ff")
    pyplot.grid()

    pyplot.xlabel("Posición en dirección E-W (m) (positiva hacia el Este)")
    pyplot.ylabel("Posición en direccion N-S (m) (positiva hacia el Norte)")
    pyplot.suptitle("Variación de las posiciones de las estaciones\n\n" + \
        "(Longitud de los vectores, exagerada en un factor 100.000;\n" + \
        "la de las barras de error, en un factor 500.000)")

    fig1.savefig("Stations_comparison.png")
    del fig1





###############
# ACTIVIDAD 3 #
###############

def exercise_3():
    """
    Representar gráficamente la deformación vertical como vectores centrados
    en cada estación, cuya dirección coincide con la dirección de la deformación
    vertical y cuyo módulo es proporcional al módulo de la deformación
    vertical. Incluir también las barras de error como estimación de la
    incertidumbre de la medida.
    """


    #>>>>>
    # Importación de módulos y paquetes.
    #>>>>>
    """
    Aquí, además del paquete 'NumPy', se necesita el paquete 'Matplotlib' (en
    concreto, el módulo 'pyplot') para realizar las representaciones gráficas.
    """
    import numpy
    from matplotlib import pyplot
    import matplotlib
    matplotlib.use("TkAgg")

    #>>>>>
    # Lectura de los archivos.
    #>>>>>
    """
    Para representar los cambios de cada punto, el programa recurrirá al archivo
    'deformation_vectors.txt'.
    """

    positions = numpy.loadtxt("data_file_1.txt", comments = "#", \
        usecols = (0, 1,))
    vectors = numpy.loadtxt("deformation_vectors.txt", comments = "#", \
        usecols = (2,))
    ends_of_vectors = positions.copy()
    ends_of_vectors[:, 1] += (vectors * 100000.0)


    #>>>>>
    # Representación gráfica de las deformaciones en horizontal.
    #>>>>>
    """
    La idea es conocer cómo han variado las posiciones de los puntos de una
    medición a la siguiente.

    Para representar gráficas 2D con barras de errores, se puede utilizar la
    funcion 'pyplot.errorbar', que permite implementar barras de error con
    tamaños personalizados para los puntos que se desee.

    Dado que el margen de error es de 1mm para todos los vectores y en
    cualquiera de las direcciones de los ejes cartesianos, sólo necesito
    proporcionar el valor '0.001' a los parámetros 'yerr' y 'xerr'... Aunque se
    han reescalado los vectores y los errores para que sean visibles.

    'ecolor' define el color de las barras de error en caso de querer un color
    diferente del color de los marcadores de los puntos. En este caso, se ha
    optado por barras azules.

    Las elipses de error se han pintado de forma manual sobre la imagen una vez
    obtenida.
    """

    fig2 = pyplot.figure(2, dpi = 200, figsize = (10, 10))

    pyplot.plot(positions[:, 0], positions[:, 1], ".r", markersize = 10)

    for i in range(len(positions)):
        pyplot.plot([positions[i, 0], ends_of_vectors[i, 0]], \
            [positions[i, 1], ends_of_vectors[i, 1]], "-b", linewidth = 4)
        pyplot.errorbar(ends_of_vectors[i, 0], ends_of_vectors[i, 1], \
            fmt = ",b", xerr = 0.0, yerr = 500.0, ecolor = "#00ff00", \
            linewidth = 2)
    pyplot.grid()

    pyplot.xlabel("Posición en dirección E-W (m) (positiva hacia el Este)")
    pyplot.ylabel("Posición en direccion N-S (m) (positiva hacia el Norte)/" + \
                  "\nDeformación del terreno")
    pyplot.suptitle("Variación de las posiciones de las estaciones\n\n" + \
        "(Longitud de los vectores, exagerada en un factor 100.000;\n" + \
        "la de las barras de error, en un factor 500.000)")

    fig2.savefig("Stations_comparison_2.png")
    del fig2





###############
# ACTIVIDAD 4 #
###############
"""
(No requiere crear un programa, sino analizar las figuras anteriormente creadas
y dar una respuesta.)
"""





###############
# ACTIVIDAD 5 #
###############
"""
Esta actividad contiene dos partes. En la primera no se necesita ningún programa
porque se pide estimar las posibles coordenadas de dos cámaras magmáticas.

En la figura, el contorno continuo representa el borde de un cráter;
las líneas discontinuas, fallas concéntricas que indican dónde puede
haber una cámara magmática profunda.
"""
def exercise_5_part_2():

    prototype_ini = numpy.loadtxt("data_file_1.txt")
    prototype_end = numpy.loadtxt("data_file_2.txt")
    prototype_deform = numpy.loadtxt("deformation_vectors.txt")

    def set_array(prototype):
        out = numpy.zeros(prototype.shape, dtype = numpy.float32)
        for i in range(prototype.shape[0]):
            for j in range(prototype.shape[1]):
                out[i, j] = prototype[i, j]
        return out

    ini = set_array(prototype_ini)
    end = set_array(prototype_end)
    deform = set_array(prototype_deform)

    out = calculations(ini, end, deform)

    e = out[0]
    s = out[1]
    d = out[2]

    output_file = open("Best_Mogi_model.py", "w")
    output_file.write("shallow_best = ")
    output_file.write(str(list(s)))
    output_file.write("\n")
    output_file.write("deep_best = ")
    output_file.write(str(list(d)))
    output_file.write("\n")
    output_file.write("error_best = ")
    output_file.write(str(e))
    output_file.close()

def calculations(ini, end, deform):
    """
    Voy a empezar por cargar las posiciones iniciales.
    """


    """
    Ahora estableceré algunos parámetros iniciales con objeto de poder aplicar
    el modelo de Mogi.

    _nu - Coeficiente de Poisson.

        Desde {1} hasta {2} en saltos de {3}:
    array_mu - Módulo de rigidez, en Pa. Se supondrá igual para ambos casos.

    array_R1 - Radio de la cámara superficial, en m.
    array_R2 - Radio de la cámara profunda, en m.

    array_delta_P - Incremento de presión en la cámara superficial, en Pa.
    array_delta_P2 - Incremento de presión en la cámara profunda, en Pa.

    array_x_chamber_1 - Coordenada 'x' de la cámara superficial, en m.
    array_y_chamber_1 - Coordenada 'y' de la cámara superficial, en m.

    array_x_chamber_2 - Coordenada 'x' de la cámara profunda, en m.
    array_y_chamber_2 - Coordenada 'y' de la cámara profunda, en m.

    array_depth - Profundidad de la cámara superficial, en m.
        La de la cámara profunda es el doble.

    ACTUALIZACIÓN. Unos intentos preliminares, ligeramente diferentes entre sí
    en lo que respecta a los intervalos de valores en los que se permitía la
    búsqueda, ha permitido dar con la localización de las cámaras magmáticas.
    A continuación, es hora de obtener los radios, los cambios de presión y
    las profundidades.
    """
    _nu = 0.25

    def set_array(ini, end, step):
        prototype_out = numpy.arange(ini, end, step)
        out = numpy.zeros((len(prototype_out),), dtype = numpy.float32)
        for i in range(len(out)):
            out[i] = prototype_out[i]
        return out

    array_x_chamber_1 = set_array(-1000.0, 2001.0, 200.0)
    array_y_chamber_1 = set_array(9400.0, 11401.0, 200.0)
#    array_x_chamber_1 = numpy.array([1550.0])
#    array_y_chamber_1 = numpy.array([10750.0])

    array_x_chamber_2 = set_array(2400.0, 4401.0, 200.0)
    array_y_chamber_2 = set_array(9000.0, 11001.0, 200.0)
#    array_x_chamber_2 = numpy.array([3550.0])
#    array_y_chamber_2 = numpy.array([9450.0])

    array_R1 = set_array(200.0, 4001.0, 200.0)
    array_R2 = set_array(400.0, 15001.0, 400.0)
#    array_R1 = numpy.array([900.0])
#    array_R2 = numpy.array([1500.0])

    array_depth = set_array(200.0, 6000.1, 200.0)

    array_delta_log10V = set_array(3.0, 8.001, 0.1)

    """
    Para ir guardando cuál es el mejor modelo, se usarán dos diccionarios,
    uno para la cámara superficial y otro para la profunda.
    """

    """
    Para decidir si el nuevo modelo es mejor, se usará una variable que guardará
    el error cuadrático medio.
    """
    error_best = 1.0 * math.pow(10, 20)

    """
    También necesito, cómo no, otros dos para guardar el modelo que se está
    probando en cada iteración.
    """
    error_current = 0.0

    """
    Para saber cuánto le podría quedar al programa para completar esta función,
    voy a calcular el total de iteraciones que tendrá que hacer.
    """
    l_r1 = len(array_R1)
    l_r2 = len(array_R2)
    l_d = len(array_depth)
    l_dV = len(array_delta_log10V)
    l_x1 = len(array_x_chamber_1)
    l_y1 = len(array_y_chamber_1)
    l_x2 = len(array_x_chamber_2)
    l_y2 = len(array_y_chamber_2)

    number_of_models = l_r1 * l_r2 * l_d * l_dV * \
        l_x1 * l_y1 * l_x2 * l_y2

    c1c2 = l_x1 * l_y1 * l_x2 * l_y2
    print("Posiciones de ambas cámaras:", c1c2)
    dc1c2 = l_d * c1c2
    print("Cámaras y profundidades:", dc1c2)

    import pyopencl
    import pyopencl.array as pyopencl_array

    platform = pyopencl.get_platforms()[0]
    context = pyopencl.Context(platform.get_devices())
    queue = pyopencl.CommandQueue(context)
    memory_flags = pyopencl.mem_flags

    kernels_file = open("kernels.cl", "r")
    kernels = pyopencl.Program(context, kernels_file.read()).build()
    kernels_file.close()

    completed = 0

    """
    Ahora se implementarán los bucles anidados.
    Esto hay que hacerlo con cuidado.

    Datos de cada array (en orden):
    - Cambio de volumen
    - Radio
    - Profundidad
    - Coordenada X
    - Coordenada Y

    Orden de anidación:
    - Cambio de volumen
    - Radio1
    - Radio2
    - Profundidad
    - Coordenada X1
    - Coordenada X2
    - Coordenada Y1
    - Coordenada Y2
    """
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # These arrays will keep track of the current model that is tested and the best model so far.
    #
    shallow_best = numpy.zeros((5,), dtype = numpy.float32)
    deep_best = numpy.zeros((5,), dtype = numpy.float32)
    shallow_current = numpy.zeros((5,), dtype = numpy.float32)
    deep_current = numpy.zeros((5,), dtype = numpy.float32)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # This array will store all the error values returned by the models.
    #
    errors_array_size = l_d * l_x1 * l_y1 * l_x2 * l_y2
    errors_array_dimensions = numpy.zeros((5,), dtype = numpy.int32)
    errors_array_dimensions[0] = l_d
    errors_array_dimensions[1] = l_x1
    errors_array_dimensions[2] = l_y1
    errors_array_dimensions[3] = l_x2
    errors_array_dimensions[4] = l_y2
    errors_array = numpy.zeros((errors_array_size,), dtype = numpy.float32)

    # Now I'll send it to the GPU.
    errors_array_gpu = pyopencl_array.to_device(queue, errors_array)
    dimensions_gpu = pyopencl_array.to_device(queue, errors_array_dimensions)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # I also need to send the 'ini' and 'deform' arrays to the GPU.
    #
    ini_shape = numpy.array(ini.shape)
    ini_shape_gpu = pyopencl_array.to_device(queue, ini_shape)
    ini_flat_gpu = pyopencl_array.to_device(queue, ini.flatten())
    deform_flat_gpu = pyopencl_array.to_device(queue, deform.flatten())

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # I need the arrays for the possible X and Y positions of both chambers
    # as well as the depth array.
    #
    array_x_chamber_1_gpu = pyopencl_array.to_device(queue, array_x_chamber_1)
    array_y_chamber_1_gpu = pyopencl_array.to_device(queue, array_y_chamber_1)
    array_x_chamber_2_gpu = pyopencl_array.to_device(queue, array_x_chamber_2)
    array_y_chamber_2_gpu = pyopencl_array.to_device(queue, array_y_chamber_2)
    array_depth_gpu = pyopencl_array.to_device(queue, array_depth)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Another array must be ready to be sent to the GPU.
    # Its content: the volume transfer, the radii of chambers 1 and 2, and 'nu'.
    #
    features = numpy.zeros((4,), dtype = numpy.float32)
    features[3] = _nu

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # This is just a counter that lets the user know how long until the programme ends the loops.
    #
    counter_delta = errors_array_size
    for DeltaV in array_delta_log10V:
        features[0] = DeltaV
        shallow_current[0] = math.pow(10, DeltaV)
        deep_current[0] = (-1) * shallow_current[0]
        for R1 in array_R1:
            features[1] = R1
            shallow_current[1] = R1
            for R2 in array_R2:
                features[2] = R2
                deep_current[1] = R2

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # I can now send the 'features' array to the GPU.
                #
                features_gpu = pyopencl_array.to_device(queue, features)

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # Calculations in GPU - 1 - Find plausible deformation models and calculate the deformation.
                # Requires: features_gpu, array_depth_gpu, array_x_chamber_1_gpu, array_y_chamber_1_gpu, \
                #     array_x_chamber_2_gpu, array_y_chamber_2_gpu, ini_flat_gpu, deform_flat_gpu, \
                #     ini_shape_gpu, dimensions_gpu, errors_array_gpu.
                #
                kernels.plausible_def_models(queue, (errors_array_size,), None, \
                    features_gpu.data, array_depth_gpu.data, array_x_chamber_1_gpu.data, \
                    array_y_chamber_1_gpu.data, array_x_chamber_2_gpu.data, array_y_chamber_2_gpu.data, \
                    ini_flat_gpu.data, deform_flat_gpu.data, ini_shape_gpu.data, dimensions_gpu.data, \
                    errors_array_gpu.data, errors_array_gpu.data)

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # Calculations in GPU - 2 - Fetch resulting errors array.
                #
                errors = errors_array_gpu.get()

                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # This is meant to be replaced by the GPU implementation.
                #
#                for index_1 in numba.prange(l_d):
#                    shallow_current[2] = array_depth[index_1]
#                    deep_current[2] = shallow_current[2] * 2.0
#                    for index_2 in numba.prange(l_x1):
#                        shallow_current[3] = array_x_chamber_1[index_2]
#                        for index_3 in numba.prange(l_y1):
#                            shallow_current[4] = array_y_chamber_1[index_3]
#                            for index_4 in numba.prange(l_x2):
#                                deep_current[3] = array_x_chamber_2[index_4]
#                                for index_5 in numba.prange(l_y2):
#                                    deep_current[4] = array_y_chamber_2[index_5]
#                                    # Obtención del error entre predicción y observación.
#                                    if Auxiliary_module.check_function(shallow_current, deep_current):
#                                        a = Auxiliary_module.deformation(\
#                                            ini, \
#                                            shallow_current, \
#                                            deep_current)
#                                        errors_array[index_1][index_2][index_3][index_4][index_5] = \
#                                            Auxiliary_module.error(deform, a)
#                                    # En caso de configuración errónea, se asigna un error muy grande para
#                                    # evitar escoger esta configuración.
#                                    else:
#                                        errors_array[index_1][index_2][index_3][index_4][index_5] = \
#                                            math.pow(10.0, 100.0)
                #
                # End of fragment that is intended to be replaced by GPU implementation.
                # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                a = min(errors)

                if a < error_best:
                    error_best = a
                    ind = numpy.unravel_index(numpy.argmin(errors), errors_array_dimensions)
                    shallow_best[0] = shallow_current[0]
                    deep_best[0] = deep_current[0]
                    shallow_best[1] = shallow_current[1]
                    deep_best[1] = deep_current[1]
                    shallow_best[2] = array_depth[ind[0]]
                    deep_best[2] = shallow_best[2] * 2.0
                    shallow_best[3] = array_x_chamber_1[ind[1]]
                    shallow_best[4] = array_y_chamber_1[ind[2]]
                    deep_best[3] = array_x_chamber_2[ind[3]]
                    deep_best[4] = array_y_chamber_2[ind[4]]
                    print("shallow_best", shallow_best)
                    print("deep_best", deep_best)
                    print("error_best", error_best)
                completed += counter_delta
                print(number_of_models, "/", completed)

    print("shallow_best", shallow_best)
    print("deep_best", deep_best)
    print("error_best", error_best)
    return (error_best, shallow_best, deep_best)

#>>>>>
# EXTRA. Representación gráfica de las deformaciones según el mejor modelo.
#>>>>>
def best_model_results():
    """
    ¿De qué sirven los datos del modelo si no vemos una gráfica de lo que se deriva
    del mismo?
    """
    import numpy
    from matplotlib import pyplot
    from Best_Mogi_model import shallow_best, deep_best
    import Auxiliary_module
    import matplotlib
    matplotlib.use("TkAgg")

    data_array_1 = numpy.loadtxt("data_file_1.txt", comments = "#")
    theor_deform = Auxiliary_module.deformation(data_array_1, \
        numpy.array(shallow_best), numpy.array(deep_best))
    positions = data_array_1[:, 0:2]
    vectors = theor_deform[:, 0:2]
    ends_of_vectors = positions + (vectors * 100000.0)

    """
    Representación de las deformaciones teóricas en el plano XY.
    """
    fig3 = pyplot.figure(3, dpi = 200, figsize = (10, 10))

    pyplot.plot(positions[:, 0], positions[:, 1], ".r")

    for i in range(len(positions)):
        pyplot.errorbar([positions[i, 0], ends_of_vectors[i, 0]], \
            [positions[i, 1], ends_of_vectors[i, 1]], fmt = "-b", \
            xerr = [0.0, 500.0], yerr = [0.0, 500.0], \
            ecolor = "#ff00ff")
    pyplot.grid()

    pyplot.xlabel("Posición en dirección E-W (m) (positiva hacia el Este)")
    pyplot.ylabel("Posición en direccion N-S (m) (positiva hacia el Norte)")
    pyplot.suptitle("Variación de las posiciones de las estaciones\n" + \
        "según el mejor modelo que se ha calculado\n\n" + \
        "(Longitud de los vectores, exagerada en un factor 100.000;\n" + \
        "la de las barras de error, en un factor 500.000)")

    fig3.savefig("Best_model_XY_deformation.png")
    del fig3

    """
    Representación de las deformaciones teóricas en dirección vertical.
    """

    vectors = theor_deform[:, 2]
    ends_of_vectors = positions.copy()
    ends_of_vectors[:, 1] += (vectors * 100000.0)

    fig4 = pyplot.figure(4, dpi = 200, figsize = (10, 10))

    pyplot.plot(positions[:, 0], positions[:, 1], ".r", markersize = 10)

    for i in range(len(positions)):
        pyplot.plot([positions[i, 0], ends_of_vectors[i, 0]], \
            [positions[i, 1], ends_of_vectors[i, 1]], "-b", linewidth = 4)
        pyplot.errorbar(ends_of_vectors[i, 0], ends_of_vectors[i, 1], \
            fmt = ",b", xerr = 0.0, yerr = 500.0, ecolor = "#00ff00", \
            linewidth = 2)
    pyplot.grid()

    pyplot.xlabel("Posición en dirección E-W (m) (positiva hacia el Este)")
    pyplot.ylabel("Posición en direccion N-S (m) (positiva hacia el Norte)/" + \
                  "\nDeformación del terreno")
    pyplot.suptitle("Variación de las posiciones de las estaciones\n\n" + \
        "(Longitud de los vectores, exagerada en un factor 100.000;\n" + \
        "la de las barras de error, en un factor 500.000)")

    fig4.savefig("Best_model_vertical_deformation.png")
    del fig4
