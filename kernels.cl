/* Función de cálculo de volumen a partir del radio. */
float volume(float radius)
{
    float a = (4.0f / 3.0f);
    float res = a * M_PI_F * pow(radius, 3.0f);
    return res;
}

/* Función de cálculo de variación del radio para cámara profunda. */
float delta_radius_2(float delta_v, float radius)
{
    float v_end = volume(radius) - delta_v;
    float r_end = pow((v_end * 0.75 / M_PI_F), (1.0 / 3.0));
    float res = r_end - radius;
    return res;
}

/* Función de cálculo de variación del radio para cámara superficial. */
float delta_radius_1(float delta_v, float radius)
{
    float v_end = volume(radius) + delta_v;
    float r_end = pow((v_end * 0.75 / M_PI_F), (1.0 / 3.0));
    float res = r_end - radius;
    return res;
}

/* Determinación de modelos plausibles.

Información que necesito:
> Array con las posibles profundidades.
> Array con las posibles coordenadas X de la cámara 1.
> Ídem con la coordenada Y.
> Ídem para la cámara 2.
> Array con las posiciones iniciales.
> Array con las deformaciones.

Contenedor de los resultados: array de tipo 'double'.
> Dimensiones: 1
> Número de elementos: número de posiciones posibles de las cámaras * número de profundidades posibles. */

__kernel void plausible_def_models(__global const float *features, __global const float *depth, \
__global const float *p_x1, __global const float *p_y1, __global const float *p_x2, __global const float *p_y2, \
__global const float *arr_ini, __global const float *arr_def, __global const int *arr_ini_shape, \
__global const int *arr_err_dim, __global const float *arr_err_in, __global float *arr_err_out)
{
    /* Get item's ID. */
    int gid0 = get_global_id(0);

    /* Obtain item's corresponding indices for the flattened errors array. */
    int index_d = gid0 / (arr_err_dim[0] * arr_err_dim[1] * arr_err_dim[2] * arr_err_dim[3]);
    int gid1 = gid0 % arr_err_dim[0];
    int index_x1 = gid1 / arr_err_dim[1];
    int gid2 = gid1 % arr_err_dim[1];
    int index_y1 = gid2 / arr_err_dim[2];
    int gid3 = gid2 % arr_err_dim[2];
    int index_x2 = gid3 / arr_err_dim[3];
    int index_y2 = gid3 % arr_err_dim[3];


    /* Do the parameters make sense?
       We check:
       > Whether the distance between the centres of the chambers is, at least,
         equal to the sum of their radii.
       > Whether the depth of the shallow chamber (below sea level) + 1000.0
         is less than any of the radii.
       > Whether the shallow chamber is smaller than the deep one.
       > Whether, after the mass transfer, the deep chamber's radius stays greater than zero.*/
    if (islessequal((depth[index_d] + 1000.0), features[1]))
    {
        arr_err_out[gid0] = 1000000.0;
    }
    else if (islessequal((depth[index_d] + 1000.0), features[2]))
    {
        arr_err_out[gid0] = 1000000.0;
    }
    else if (islessequal(features[2], features[1]))
    {
        arr_err_out[gid0] = 1000000.0;
    }
    else
    {
        float check_1a = pow((p_x1[index_x1] - p_x2[index_x2]), 2.0);
        float check_1b = pow((p_y1[index_y1] - p_y2[index_y2]), 2.0);
        float check_1c = pow(depth[index_d], 2.0);
        float check_1 = sqrt(check_1a + check_1b + check_1c);

        if (islessequal(check_1, (features[1] + features[2])))
        {
            arr_err_out[gid0] = 1000000.0;
        }
        else
        {

            /* Initialise the required variables:
               > One for the total error.
               > Six for deformation contributions in each direction by each chamber.
               > One for the error contribution by the position in question.
               > Three for the point's final coordinates.
               > Some variables that are required for the Mogi model.
            */
            float total = 0.0;
            float def1x = 0.0;
            float def1y = 0.0;
            float def1z = 0.0;
            float def2x = 0.0;
            float def2y = 0.0;
            float def2z = 0.0;
            float subtotal = 0.0;
            float end_x = 0.0;
            float end_y = 0.0;
            float end_z = 0.0;
            float rel_depth_1 = 0.0;
            float rel_depth_2 = 0.0;
            int int_aux_x = 0;
            int int_aux_y = 0;
            int int_aux_z = 0;
            float float_aux_1 = 0.0;
            float float_aux_2 = 0.0;

            /* FIXME- Let's go one position at a time. */
            for (int i = 0; i < arr_ini_shape[0]; i++)
            {
                /* Reset the contribution. */
                subtotal = 0.0;

                /* FIXED - Mogi model - Relative depth. */
                int_aux_x = i * arr_ini_shape[1];
                int_aux_y = int_aux_x + 1;
                int_aux_z = int_aux_x + 2;
                rel_depth_1 = depth[index_d] + arr_ini[int_aux_z];
                rel_depth_2 = 2.0 * depth[index_d] + arr_ini[int_aux_z];

                /* Mogi model - First calculation of deformation. */
                def1z = (1.0 - features[3]) * pow(features[1], 3.0) * (delta_radius_1(features[0], features[1]) / features[1]) * 4.0 * rel_depth_1;
                def2z = (1.0 - features[3]) * pow(features[2], 3.0) * (delta_radius_2(features[0], features[2]) / features[2]) * 4.0 * rel_depth_2;
                def1x = arr_ini[int_aux_x] - p_x1[index_x1];
                def1y = arr_ini[int_aux_y] - p_y1[index_y1];
                def2x = arr_ini[int_aux_x] - p_x2[index_x2];
                def2y = arr_ini[int_aux_y] - p_y2[index_y2];

                /* Mogi model - 'r'. */
                float_aux_1 = sqrt(pow(def1x, 2.0) + pow(def1y, 2.0));
                float_aux_2 = sqrt(pow(def2x, 2.0) + pow(def2y, 2.0));

                /* Mogi model - Fix deformations. */
                def1z = def1z / pow(pow(rel_depth_1, 2.0) + pow(float_aux_1, 2.0), 1.5);
                def2z = def2z / pow(pow(rel_depth_2, 2.0) + pow(float_aux_2, 2.0), 1.5);

                float_aux_1 = def1z / rel_depth_1;
                float_aux_2 = def2z / rel_depth_2;

                def1x = float_aux_1 * def1x;
                def2x = float_aux_2 * def2x;
                def1y = float_aux_1 * def1y;
                def2y = float_aux_2 * def2y;

                /* Add deformation contributions. */
                end_x = def1x + def2x;
                end_y = def1y + def2y;
                end_z = def1z + def2z;

                /* Compare theoretical deformations to experimental deformations. */
                subtotal += pow(end_x - arr_def[int_aux_x], 2.0);
                subtotal += pow(end_y - arr_def[int_aux_y], 2.0);
                subtotal += pow(end_z - arr_def[int_aux_z], 2.0);

                /* Add to 'total'. */
                total += sqrt(subtotal);
            }

            /* Calculate 'sqrt' of 'total', and place it in the appropriate cell. */
            arr_err_out[gid0] = total / arr_ini_shape[0];
        }
    }
}


/* Cálculo de los errores.

Si alguna celda del array de errores es igual a cero, el modelo asociado se entiende como factible
y se puede proceder a calcular el error cometido.
*/

__kernel void def_models(__global const float *features, __global const float *depth, \
__global const float *p_x1, __global const float *p_y1, __global const float *p_x2, __global const float *p_y2, \
__global const float *arr_ini, __global const float *arr_def, __global const int *arr_ini_shape, \
__global const int *arr_err_dim, __global const float *arr_err_in, __global float *arr_err_out)
{
    /* Get item's ID. */
    int gid0 = get_global_id(0);

    /* FIXED - Obtain item's corresponding indices for the flattened errors array. */
    int index_d = gid0 / (arr_err_dim[4] * arr_err_dim[1] * arr_err_dim[2] * arr_err_dim[3]);
    int gid1 = gid0 % (arr_err_dim[4] * arr_err_dim[1] * arr_err_dim[2] * arr_err_dim[3]);
    int index_x1 = gid1 / (arr_err_dim[4] * arr_err_dim[2] * arr_err_dim[3]);
    int gid2 = gid1 % (arr_err_dim[4] * arr_err_dim[2] * arr_err_dim[3]);
    int index_y1 = gid2 / (arr_err_dim[4] * arr_err_dim[3]);
    int gid3 = gid2 % (arr_err_dim[4] * arr_err_dim[3]);
    int index_x2 = gid3 / arr_err_dim[4];
    int index_y2 = gid3 % arr_err_dim[4];

    arr_err_out[gid0] = arr_err_in[gid0];

    if (isless(arr_err_in[gid0], 1.0))
    {
        /* Initialise the required variables:
           > One for the total error.
           > Six for deformation contributions in each direction by each chamber.
           > One for the error contribution by the position in question.
           > Three for the point's final coordinates.
           > Some variables that are required for the Mogi model.
        */
        float total = 0.0;
        float def1x = 0.0;
        float def1y = 0.0;
        float def1z = 0.0;
        float def2x = 0.0;
        float def2y = 0.0;
        float def2z = 0.0;
        float subtotal = 0.0;
        float end_x = 0.0;
        float end_y = 0.0;
        float end_z = 0.0;
        float rel_depth_1 = 0.0;
        float rel_depth_2 = 0.0;
        int int_aux_x = 0;
        int int_aux_y = 0;
        int int_aux_z = 0;
        float float_aux_1 = 0.0;
        float float_aux_2 = 0.0;

        arr_err_out[gid0] = 1.0;

        /* FIXME- Let's go one position at a time. */
        for (int i = 0; i < arr_ini_shape[0]; i++)
        {
            /* Reset the contribution. */
            subtotal = 0.0;

            /* FIXED - Mogi model - Relative depth. */
            int_aux_x = i * arr_ini_shape[1];
            int_aux_y = int_aux_x + 1;
            int_aux_z = int_aux_x + 2;
            rel_depth_1 = depth[index_d] + arr_ini[int_aux_z];
            rel_depth_2 = 2.0 * depth[index_d] + arr_ini[int_aux_z];

            /* Mogi model - First calculation of deformation. */
            def1z = (1.0 - features[3]) * pow(features[1], 3.0) * (delta_radius_1(features[0], features[1]) / features[1]) * 4.0 * rel_depth_1;
            def2z = (1.0 - features[3]) * pow(features[2], 3.0) * (delta_radius_2(features[0], features[2]) / features[2]) * 4.0 * rel_depth_2;
            def1x = arr_ini[int_aux_x] - p_x1[index_x1];
            def1y = arr_ini[int_aux_y] - p_y1[index_y1];
            def2x = arr_ini[int_aux_x] - p_x2[index_x2];
            def2y = arr_ini[int_aux_y] - p_y2[index_y2];

            /* Mogi model - 'r'. */
            float_aux_1 = sqrt(pow(def1x, 2.0) + pow(def1y, 2.0));
            float_aux_2 = sqrt(pow(def2x, 2.0) + pow(def2y, 2.0));

            /* Mogi model - Fix deformations. */
            def1z = def1z / pow(pow(rel_depth_1, 2.0) + pow(float_aux_1, 2.0), 1.5);
            def2z = def2z / pow(pow(rel_depth_2, 2.0) + pow(float_aux_2, 2.0), 1.5);

            float_aux_1 = def1z / rel_depth_1;
            float_aux_2 = def2z / rel_depth_2;

            def1x = float_aux_1 * def1x;
            def2x = float_aux_2 * def2x;
            def1y = float_aux_1 * def1y;
            def2y = float_aux_2 * def2y;

            /* Add deformation contributions. */
            end_x = def1x + def2x;
            end_y = def1y + def2y;
            end_z = def1z + def2z;

            /* Compare theoretical deformations to experimental deformations. */
            subtotal += pow(end_x - arr_def[int_aux_x], 2.0);
            subtotal += pow(end_y - arr_def[int_aux_y], 2.0);
            subtotal += pow(end_z - arr_def[int_aux_z], 2.0);

            /* Add to 'total'. */
            total += sqrt(subtotal);
        }

        /* Calculate 'sqrt' of 'total', and place it in the appropriate cell. */
        arr_err_out[gid0] = total / arr_ini_shape[0];

    }
    else
    {
        arr_err_out[gid0] = pow(10.0, 9.0);
    }
}
