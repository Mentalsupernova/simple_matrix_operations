#ifndef SIMPLE_MATRIX_OPERATIONS_H
#define SIMPLE_MATRIX_OPERATIONS_H
#include <stdlib.h>
#include <stdio.h>
#include "stdint.h"
#include <pthread.h>
#include "string.h"
#include "math.h"

/*for CPU style multiplication*/
#define BLOCK_SIZE 128
#define NUM_THREADS 16


/*for GPU style multiplication*/
#define SHIT_TILE_SIZE (16)
/* CHANGE THIS FOR UR MACHINE */
#define NVIDIA_DOT_PPTX_KERNEL_FPATH "/home/utsu/projects/c/image_processing/libs/cuda_kernels/nvidia_gpu_dot.ptx"

typedef enum {
  RANDOM=0,
  ZEROS=1,
  ONES
}MATRIX_INITIALIZATION_TYPE;


#define MATRIX_ROWS(mat) ((mat) && (mat)->dims && (mat)->ndims >= 1 ? (mat)->dims[0] : 0)
#define MATRIX_COLS(mat) ((mat) && (mat)->dims && (mat)->ndims >= 2 ? (mat)->dims[1] : 0)



typedef struct {
    size_t ndims;       // Number of dimensions
    size_t *dims;       // Sizes of each dimension
    double *mat;        // Flat data array
} simple_matrix;




typedef struct {
    size_t start_row, end_row;
    simple_matrix *matrix_a;
    simple_matrix *matrix_b;
    simple_matrix *output;
} thread_data;


/*
 * @brief allocates matrix with predefined method for initialization
 * @line pass random min max if u choose random initialization method in othercases u can pass 0
 * @line random - allocates matrix with random numbers with random seed
 * @line for useing random initialize seed by yourself   srand((unsigned int)time(NULL));  // Seed for random numbers 
 * @line zeros - allocates matrix with zeros e.g
 * @line {0,0}
 * @line {0,0}
 * @line ones - allocates matrix with 1 e.g
 * @line {1,1}
 * @line {1,1}
 */




extern simple_matrix *filter_rows(simple_matrix *matrix, int (*condition)(double *row, size_t cols));
extern simple_matrix *extract_columns(simple_matrix *matrix, size_t *columns, size_t num_columns);
extern simple_matrix *allocate_matrix(size_t ndims, size_t *dims, uint8_t initialization_method, double random_min, double random_max);
extern void add_cpu(simple_matrix * matrix_a,simple_matrix * matrix_b,simple_matrix * output);
extern void substruct_cpu(simple_matrix * matrix_a,simple_matrix * matrix_b,simple_matrix * output);
extern void transpose(simple_matrix * matrix);
extern void inverse(simple_matrix * matrix);
extern void free_matrix(simple_matrix * matrix);
extern void print_matrix(simple_matrix * matrix);
extern void dot_parallel(simple_matrix *matrix_a, simple_matrix *matrix_b, simple_matrix *output);
extern void *dot_worker(void *arg);
extern void scale_columns(simple_matrix *matrix, double scalar);
extern void apply_function_to_column(simple_matrix *matrix, size_t col_index, double (*func)(double));
extern void scale_column(simple_matrix *matrix, size_t col_index, double scalar);
extern void add_scalar_to_column(simple_matrix *matrix, size_t col_index, double scalar);
extern double sum_column(simple_matrix *matrix, size_t col_index);
extern void normalize_columns(simple_matrix *matrix);
extern void apply_function_to_matrix(simple_matrix *matrix, double (*func)(double));
extern simple_matrix *extract_column(simple_matrix *matrix, size_t col_index);
extern void replace_column(simple_matrix *matrix, size_t col_index, simple_matrix *new_column);
extern void add_columns(simple_matrix *matrix, simple_matrix *new_columns);
extern simple_matrix *slice_rows(simple_matrix *matrix, int start, int end);
extern void print_shape(simple_matrix * mat);
extern void matrix_to_csv(simple_matrix *mat, const char *filename);
extern simple_matrix *matrix_from_csv(const char *filename);
extern simple_matrix *extract_row(simple_matrix *matrix, size_t row_index);
extern  size_t calculate_index(size_t ndims, size_t *dims, size_t *indices);
extern int reshape(simple_matrix *matrix, size_t new_ndims, size_t *new_dims);

#ifdef NVIDIA_GPU_DEVICE

#include <stdio.h>
#define TILE_SIZE 16

void dot_parallel_gpu_nvidia(double *a, double *b, double *c, size_t rows_a, size_t cols_a, size_t cols_b);
void gpu_matrix_add(double *a, double *b, double *c, size_t rows, size_t cols);
void gpu_matrix_sub(double *a, double *b, double *c, size_t rows, size_t cols);
#define dot(mat_a, mat_b, output) (dot_parallel_gpu_nvidia(mat_a->mat, mat_b->mat, output->mat, mat_a->dims[0], mat_a->dims[1], mat_b->dims[1]))
#define add(mat_a, mat_b, output) (gpu_matrix_add(mat_a->mat, mat_b->mat, output->mat, mat_a->dims[0],mat_a->dims[1]))
#define substruct(mat_a, mat_b, output) (gpu_matrix_sub(mat_a->mat, mat_b->mat, output->mat, mat_a->dims[0],mat_a->dims[1]))
#else
#define dot(mat_a,mat_b,output) (dot_parallel(mat_a,mat_b,output))
#define add(mat_a,mat_b,output) (add_cpu(mat_a,mat_b,output))
#define substruct(mat_a,mat_b,output) (substruct_cpu(mat_a,mat_b,output))
#endif

/*
 * @brief simple matrix operations lirary which has predefined methods for
 * @line MUST BE LINKED WITH CMATH LIB 
 * @line simple_matrix * allocate_matrix(size_t rows, size_t cols,MATRIX_INITIAlIZATION_TYPE initialization_method);
 * @line void add(simple_matrix * matrix_a,simple_matrix * matrix_b,simple_matrix * output); - dot product of a+b = output 
 * @line void substruct(simple_matrix * matrix_a,simple_matrix * matrix_b,simple_matrix * output); - dot product of a-b = output 
 * @line void dot(simple_matrix * matrix_a,simple_matrix * matrix_b,simple_matrix * output); - dot product of axb = output 
 * @line void transpose(simple_matrix * matrix); - provides transposed matrix as  matrix matrix pointer updates after transpose
 * @line void inverse(simple_matrix * matrix); - provides inversed matrix as  matrix matrix pointer updates after inverse
 * @line void free_matrix(simple_matrix * matrix);
 */





/*
 * SMO IMPLEMENTATION
 */
#ifdef SIMPLE_MATRIX_OPERATIONS_IMPLEMETATION


#ifdef NVIDIA_GPU_DEVICE
#endif
int reshape(simple_matrix *matrix, size_t new_ndims, size_t *new_dims) {
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndims; i++) {
        new_size *= new_dims[i];
    }

    size_t old_size = 1;
    for (size_t i = 0; i < matrix->ndims; i++) {
        old_size *= matrix->dims[i];
    }

    if (new_size != old_size) {
        printf("Error: Total size must remain the same for reshaping.\n");
        return -1;
    }

    free(matrix->dims);
    matrix->dims = (size_t *)malloc(new_ndims * sizeof(size_t));
    if (!matrix->dims) {
        printf("Error: Unable to allocate memory for new dimensions.\n");
        return -1;
    }

    matrix->ndims = new_ndims;
    memcpy(matrix->dims, new_dims, new_ndims * sizeof(size_t));
    return 0;
}

size_t calculate_index(size_t ndims, size_t *dims, size_t *indices) {
    size_t index = 0;
    size_t stride = 1;

    for (size_t i = ndims; i-- > 0;) { // Loop in reverse
        index += indices[i] * stride;
        stride *= dims[i];
    }

    return index;
}
simple_matrix *extract_row(simple_matrix *matrix, size_t row_index) {
    if (matrix == NULL || matrix->mat == NULL || row_index >= MATRIX_ROWS(matrix)) {
        printf("Error: Invalid row index or matrix.\n");
        return NULL;
    }

    // Allocate a new matrix to hold the extracted row
    simple_matrix *row = allocate_matrix(2, (size_t[]){1, MATRIX_COLS(matrix)}, ZEROS, 0.0, 0.0);
    if (!row) {
        printf("Error: Unable to allocate memory for the row.\n");
        return NULL;
    }

    // Copy the row data
    for (size_t j = 0; j < MATRIX_COLS(matrix); j++) {
        row->mat[j] = matrix->mat[row_index * MATRIX_COLS(matrix) + j];
    }

    return row;
}


simple_matrix *matrix_from_csv(const char *filename) {
    if (filename == NULL) {
        printf("Error: Invalid filename.\n");
        return NULL;
    }

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Unable to open file for reading: %s\n", filename);
        return NULL;
    }

    size_t rows = 0, cols = 0;
    char line[1024];

    while (fgets(line, sizeof(line), file)) {
        rows++;
        if (cols == 0) { 
            char *token = strtok(line, ",");
            while (token != NULL) {
                cols++;
                token = strtok(NULL, ",");
            }
        }
    }

    rewind(file);

    simple_matrix *mat = allocate_matrix(2, (size_t[]){rows, cols}, ZEROS, 0.0, 0.0);
    if (mat == NULL) {
        printf("Error: Unable to allocate memory for matrix.\n");
        fclose(file);
        return NULL;
    }

    size_t row = 0;
    while (fgets(line, sizeof(line), file)) {
        size_t col = 0;
        char *token = strtok(line, ",");
        while (token != NULL && col < cols) {
            mat->mat[row * cols + col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(file);
    printf("Matrix successfully loaded from %s\n", filename);
    return mat;
}
void matrix_to_csv(simple_matrix *mat, const char *filename) {
    if (mat == NULL || mat->mat == NULL || filename == NULL) {
        printf("Error: Invalid input to matrix_to_csv.\n");
        return;
    }

    if (mat->ndims != 2) {
        printf("Error: Only 2D matrices can be written to a CSV file.\n");
        return;
    }

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Unable to open file for writing: %s\n", filename);
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(mat); i++) {
        for (size_t j = 0; j < MATRIX_COLS(mat); j++) {
            fprintf(file, "%f", mat->mat[i * MATRIX_COLS(mat) + j]);
            if (j < MATRIX_COLS(mat) - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Matrix successfully written to %s\n", filename);
}



void print_shape(simple_matrix * mat){
  for(size_t i;i<mat->ndims;i++){

  printf("%d X ",mat->dims[i]);
  

  }
  printf("\n");
}

simple_matrix *slice_rows(simple_matrix *matrix, int start, int end) {
    if (matrix == NULL || matrix->mat == NULL) {
        printf("Error: Invalid matrix.\n");
        return NULL;
    }

    if (start < 0) start = (int)MATRIX_ROWS(matrix) + start;
    if (end < 0) end = (int)MATRIX_ROWS(matrix) + end;

    if (start < 0) start = 0;
    if (end > (int)MATRIX_ROWS(matrix)) end = (int)MATRIX_ROWS(matrix);
    if (start >= end) {
        printf("Error: Invalid slice range.\n");
        return NULL;
    }

    size_t slice_rows = (size_t)(end - start);
    size_t cols = MATRIX_COLS(matrix);
    simple_matrix *result = allocate_matrix(slice_rows, (size_t []){MATRIX_COLS(matrix),slice_rows}, ZEROS, 0.0, 0.0);
    if (!result) {
        printf("Error: Unable to allocate memory for the sliced matrix.\n");
        return NULL;
    }
    memcpy(result->mat, &matrix->mat[start * cols], slice_rows * cols * sizeof(double));

    return result;
}
void add_columns(simple_matrix *matrix, simple_matrix *new_columns) {
    if (matrix == NULL || matrix->mat == NULL || new_columns == NULL || new_columns->mat == NULL) {
        printf("Error: Invalid input matrix or columns.\n");
        return;
    }

    if (matrix->ndims != 2 || new_columns->ndims != 2) {
        printf("Error: Both matrices must be 2D to add columns.\n");
        return;
    }

    if (MATRIX_ROWS(matrix) != MATRIX_ROWS(new_columns)) {
        printf("Error: Row count of the new columns must match the original matrix.\n");
        return;
    }

    size_t new_cols = MATRIX_COLS(matrix) + MATRIX_COLS(new_columns);
    double *new_data = (double *)malloc(MATRIX_ROWS(matrix) * new_cols * sizeof(double));
    if (!new_data) {
        printf("Error: Memory allocation failed for the new matrix.\n");
        return;
    }

    // Copy old matrix data into new matrix
    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        for (size_t j = 0; j < MATRIX_COLS(matrix); j++) {
            new_data[i * new_cols + j] = matrix->mat[i * MATRIX_COLS(matrix) + j];
        }
    }

    for (size_t i = 0; i < MATRIX_ROWS(new_columns); i++) {
        for (size_t j = 0; j < MATRIX_COLS(new_columns); j++) {
            new_data[i * new_cols + (MATRIX_COLS(matrix) + j)] = new_columns->mat[i * MATRIX_COLS(new_columns) + j];
        }
    }

    free(matrix->mat);
    matrix->mat = new_data;
    matrix->dims[1] = new_cols;  
}

void replace_column(simple_matrix *matrix, size_t col_index, simple_matrix *new_column) {
    if (matrix == NULL || matrix->mat == NULL || new_column == NULL || new_column->mat == NULL) {
        printf("Error: Invalid input for replacing column.\n");
        return;
    }

    if (matrix->ndims != 2 || new_column->ndims != 2) {
        printf("Error: Both matrices must be 2D to replace a column.\n");
        return;
    }

    if (col_index >= MATRIX_COLS(matrix)) {
        printf("Error: Column index out of bounds.\n");
        return;
    }

    if (MATRIX_COLS(new_column) != 1 || MATRIX_ROWS(new_column) != MATRIX_ROWS(matrix)) {
        printf("Error: New column must have one column and the same number of rows as the target matrix.\n");
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        matrix->mat[i * MATRIX_COLS(matrix) + col_index] = new_column->mat[i * MATRIX_COLS(new_column)];
    }
}

simple_matrix *extract_column(simple_matrix *matrix, size_t col_index) {
    if (matrix == NULL || matrix->mat == NULL || col_index >= MATRIX_COLS(matrix)) {
        printf("Error: Invalid column index or matrix.\n");
        return NULL;
    }

    size_t dims[2] = {MATRIX_ROWS(matrix), 1};
    simple_matrix *column = allocate_matrix(2, dims, ZEROS, 0.0, 0.0);
    if (!column) {
        printf("Error: Unable to allocate memory for the column.\n");
        return NULL;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        column->mat[i] = matrix->mat[i * MATRIX_COLS(matrix) + col_index];
    }

    return column;
}


void apply_function_to_matrix(simple_matrix *matrix, double (*func)(double)) {
    if (matrix == NULL || matrix->mat == NULL) {
        printf("Error: Invalid matrix.\n");
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix) * MATRIX_COLS(matrix); i++) {
        matrix->mat[i] = func(matrix->mat[i]);
    }
}

void normalize_columns(simple_matrix *matrix) {
    if (matrix == NULL || matrix->mat == NULL) {
        printf("Error: Invalid matrix.\n");
        return;
    }

    for (size_t col = 0; col < MATRIX_COLS(matrix); col++) {
        double sum = 0.0, sum_sq = 0.0;
        for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
            double value = matrix->mat[i * MATRIX_COLS(matrix) + col];
            sum += value;
            sum_sq += value * value;
        }

        double mean = sum / MATRIX_ROWS(matrix);
        double variance = (sum_sq / MATRIX_ROWS(matrix)) - (mean * mean);
        double std_dev = variance > 0 ? sqrt(variance) : 1.0;

        for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
            size_t index = i * MATRIX_COLS(matrix) + col;
            matrix->mat[index] = (matrix->mat[index] - mean) / std_dev;
        }
    }
}

double sum_column(simple_matrix *matrix, size_t col_index) {
    if (matrix == NULL || matrix->mat == NULL || col_index >= MATRIX_COLS(matrix)) {
        printf("Error: Invalid column index or matrix.\n");
        return 0.0;
    }

    double sum = 0.0;
    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        sum += matrix->mat[i * MATRIX_COLS(matrix) + col_index];
    }
    return sum;
}

void add_scalar_to_column(simple_matrix *matrix, size_t col_index, double scalar) {
    if (matrix == NULL || matrix->mat == NULL || col_index >= MATRIX_COLS(matrix)) {
        printf("Error: Invalid column index or matrix.\n");
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        matrix->mat[i * MATRIX_COLS(matrix) + col_index] += scalar;
    }
}

void scale_column(simple_matrix *matrix, size_t col_index, double scalar) {
    if (matrix == NULL || matrix->mat == NULL || col_index >= MATRIX_COLS(matrix)) {
        printf("Error: Invalid column index or matrix.\n");
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        matrix->mat[i * MATRIX_COLS(matrix) + col_index] *= scalar;
    }
}

void apply_function_to_column(simple_matrix *matrix, size_t col_index, double (*func)(double)) {
    if (matrix == NULL || matrix->mat == NULL || col_index >= MATRIX_COLS(matrix)) {
        printf("Error: Invalid column index or matrix.\n");
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        size_t index = i * MATRIX_COLS(matrix) + col_index;
        matrix->mat[index] = func(matrix->mat[index]);
    }
}

void scale_columns(simple_matrix *matrix, double scalar) {
    if (matrix == NULL || matrix->mat == NULL) {
        printf("Error: Invalid matrix.\n");
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix) * MATRIX_COLS(matrix); i++) {
        matrix->mat[i] *= scalar;
    }
}
simple_matrix *extract_columns(simple_matrix *matrix, size_t *columns, size_t num_columns) {
    if (matrix == NULL || matrix->mat == NULL || columns == NULL || num_columns == 0) {
        printf("Error: Invalid input to extract_columns.\n");
        return NULL;
    }

    size_t rows = MATRIX_ROWS(matrix);
    size_t cols = MATRIX_COLS(matrix);
    if (rows == 0 || cols == 0) {
        printf("Error: Matrix has invalid dimensions.\n");
        return NULL;
    }

    simple_matrix *result = allocate_matrix(2, (size_t[]){rows, num_columns}, ZEROS, 0.0, 0.0);
    if (!result) {
        printf("Error: Unable to allocate memory for the result matrix.\n");
        return NULL;
    }

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < num_columns; j++) {
            size_t col_index = columns[j];
            if (col_index >= cols) {
                printf("Error: Column index out of bounds (%zu >= %zu).\n", col_index, cols);
                free_matrix(result);
                return NULL;
            }
            result->mat[i * num_columns + j] = matrix->mat[i * cols + col_index];
        }
    }

    return result;
}

void *dot_worker(void *arg) {
    thread_data *data = (thread_data *)arg;

    size_t block_size = BLOCK_SIZE; 
    size_t matrix_a_cols = MATRIX_COLS(data->matrix_a);
    size_t matrix_b_cols = MATRIX_COLS(data->matrix_b);
    size_t output_cols = MATRIX_COLS(data->output);

    for (size_t bi = data->start_row; bi < data->end_row; bi += block_size) {
        for (size_t bj = 0; bj < matrix_b_cols; bj += block_size) {
            for (size_t bk = 0; bk < matrix_a_cols; bk += block_size) {
                for (size_t i = bi; i < bi + block_size && i < data->end_row; i++) {
                    for (size_t j = bj; j < bj + block_size && j < matrix_b_cols; j++) {
                        double sum = 0.0;
                        for (size_t k = bk; k < bk + block_size && k < matrix_a_cols; k++) {
                            sum += data->matrix_a->mat[i * matrix_a_cols + k] *
                                   data->matrix_b->mat[k * matrix_b_cols + j];
                        }
                        data->output->mat[i * output_cols + j] += sum;
                    }
                }
            }
        }
    }
    return NULL;
}

void dot_parallel(simple_matrix *matrix_a, simple_matrix *matrix_b, simple_matrix *output) {
    if (MATRIX_COLS(matrix_a) != MATRIX_ROWS(matrix_b)) {
        printf("Error: Matrix dimensions must match for multiplication.\n");
        return;
    }

    if (MATRIX_ROWS(output) != MATRIX_ROWS(matrix_a) || MATRIX_COLS(output) != MATRIX_COLS(matrix_b)) {
        printf("Error: Output matrix has incorrect dimensions.\n");
        return;
    }

    pthread_t threads[NUM_THREADS];
    thread_data thread_args[NUM_THREADS];

    size_t rows_per_thread = MATRIX_ROWS(matrix_a) / NUM_THREADS;
    size_t remaining_rows = MATRIX_ROWS(matrix_a) % NUM_THREADS;

    size_t start_row = 0;
    for (size_t t = 0; t < NUM_THREADS; t++) {
        size_t end_row = start_row + rows_per_thread + (t < remaining_rows ? 1 : 0);

        thread_args[t].start_row = start_row;
        thread_args[t].end_row = end_row;
        thread_args[t].matrix_a = matrix_a;
        thread_args[t].matrix_b = matrix_b;
        thread_args[t].output = output;

        pthread_create(&threads[t], NULL, dot_worker, &thread_args[t]);
        start_row = end_row;
    }

    for (size_t t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }
}




simple_matrix *allocate_matrix(size_t ndims, size_t *dims, uint8_t initialization_method, double random_min, double random_max) {
    if (ndims == 0 || dims == NULL) {
        printf("Error: Invalid dimensions provided.\n");
        return NULL;
    }

    simple_matrix *matrix = (simple_matrix *)malloc(sizeof(simple_matrix));
    if (!matrix) {
        printf("Error: Unable to allocate memory for matrix struct.\n");
        return NULL;
    }

    matrix->ndims = ndims;
    matrix->dims = (size_t *)malloc(ndims * sizeof(size_t));
    if (!matrix->dims) {
        printf("Error: Unable to allocate memory for matrix dimensions.\n");
        free(matrix);
        return NULL;
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < ndims; i++) {
        if (dims[i] == 0) {
            printf("Error: Dimensions must be greater than zero.\n");
            free(matrix->dims);
            free(matrix);
            return NULL;
        }
        matrix->dims[i] = dims[i];
        total_elements *= dims[i];
    }

    matrix->mat = (double *)malloc(total_elements * sizeof(double));
    if (!matrix->mat) {
        printf("Error: Unable to allocate memory for matrix data.\n");
        free(matrix->dims);
        free(matrix);
        return NULL;
    }

    for (size_t i = 0; i < total_elements; i++) {
        switch (initialization_method) {
            case RANDOM:
                matrix->mat[i] = random_min + ((double)rand() / RAND_MAX) * (random_max - random_min);
                break;
            case ONES:
                matrix->mat[i] = 1.0;
                break;
            case ZEROS:
            default:
                matrix->mat[i] = 0.0;
                break;
        }
    }

    return matrix;
}



void add_cpu(simple_matrix *matrix_a, simple_matrix *matrix_b, simple_matrix *output) {
    if (matrix_a->ndims != matrix_b->ndims || matrix_a->ndims != output->ndims) {
        printf("Error: Matrices must have the same number of dimensions for addition.\n");
        return;
    }

    for (size_t i = 0; i < matrix_a->ndims; i++) {
        if (matrix_a->dims[i] != matrix_b->dims[i] || matrix_a->dims[i] != output->dims[i]) {
            printf("Error: Matrices must have the same shape for addition.\n");
            return;
        }
    }

    size_t total_size = 1;
    for (size_t i = 0; i < matrix_a->ndims; i++) {
        total_size *= matrix_a->dims[i];
    }

    for (size_t i = 0; i < total_size; i++) {
        output->mat[i] = matrix_a->mat[i] + matrix_b->mat[i];
    }
}
void substruct_cpu(simple_matrix *matrix_a, simple_matrix *matrix_b, simple_matrix *output) {
    // Check input validity
    if (matrix_a == NULL || matrix_b == NULL || output == NULL || 
        matrix_a->mat == NULL || matrix_b->mat == NULL || output->mat == NULL) {
        printf("Error: Invalid input matrices.\n");
        return;
    }

    if (matrix_a->ndims != matrix_b->ndims || matrix_a->ndims != output->ndims) {
        printf("Error: Number of dimensions must match for all matrices.\n");
        return;
    }

    for (size_t d = 0; d < matrix_a->ndims; d++) {
        if (matrix_a->dims[d] != matrix_b->dims[d] || matrix_a->dims[d] != output->dims[d]) {
            printf("Error: Dimension sizes must match for subtraction.\n");
            return;
        }
    }

    size_t total_elements = 1;
    for (size_t d = 0; d < matrix_a->ndims; d++) {
        total_elements *= matrix_a->dims[d];
    }

    for (size_t i = 0; i < total_elements; i++) {
        output->mat[i] = matrix_a->mat[i] - matrix_b->mat[i];
    }
}




void transpose(simple_matrix * matrix){
    if (matrix == NULL || matrix->mat == NULL) {
        printf("Error: Invalid matrix.\n");
        exit(2);
    }
    double *transposed = (double *)malloc(MATRIX_ROWS(matrix) * MATRIX_COLS(matrix) * sizeof(double));
    if (!transposed) {
        printf("Error: Memory allocation failed.\n");
        exit(2);
    }
    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        for (size_t j = 0; j < MATRIX_COLS(matrix); j++) {
            transposed[j * MATRIX_ROWS(matrix) + i] = matrix->mat[i * MATRIX_COLS(matrix) + j];
        }
    }
    size_t temp = MATRIX_ROWS(matrix);
    matrix->dims[0] = MATRIX_COLS(matrix);
    matrix->dims[1] = temp;
    free(matrix->mat);
    matrix->mat = transposed;
}


void inverse(simple_matrix *matrix) {
    if (MATRIX_ROWS(matrix) != MATRIX_COLS(matrix)) {
        printf("Error: Only square matrices can be inverted.\n");
        exit(2);
    }

    size_t n = MATRIX_ROWS(matrix);

    double *augmented = (double *)malloc(n * 2 * n * sizeof(double));
    if (!augmented) {
        printf("Error: Memory allocation failed.\n");
        exit(2);
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            augmented[i * 2 * n + j] = matrix->mat[i * n + j];        // Original matrix
            augmented[i * 2 * n + (n + j)] = (i == j) ? 1.0 : 0.0;   // Identity matrix
        }
    }

    for (size_t i = 0; i < n; i++) {
        if (augmented[i * 2 * n + i] == 0.0) {
            printf("Error: Matrix is singular and cannot be inverted.\n");
            free(augmented);
            exit(2);
        }

        double pivot = augmented[i * 2 * n + i];
        for (size_t j = 0; j < 2 * n; j++) {
            augmented[i * 2 * n + j] /= pivot;
        }

        for (size_t k = 0; k < n; k++) {
            if (k == i) continue; // 
            double factor = augmented[k * 2 * n + i];
            for (size_t j = 0; j < 2 * n; j++) {
                augmented[k * 2 * n + j] -= factor * augmented[i * 2 * n + j];
            }
        }
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            matrix->mat[i * n + j] = augmented[i * 2 * n + (n + j)];
        }
    }

    free(augmented);
}
void free_matrix(simple_matrix * matrix){
  free(matrix->mat);
  free(matrix);
}

void print_matrix(simple_matrix *matrix) {
    if (matrix == NULL || matrix->mat == NULL) {
        printf("Error: Invalid matrix.\n");
        return;
    }

    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        for (size_t j = 0; j < MATRIX_COLS(matrix); j++) {
            printf(" %f", matrix->mat[i * MATRIX_COLS(matrix) + j]);
        }
        printf("\n");
    }
}

simple_matrix *filter_rows(simple_matrix *matrix, int (*condition)(double *row, size_t cols)) {
    if (matrix == NULL || matrix->mat == NULL || condition == NULL) {
        printf("Error: Invalid input to filter_rows.\n");
        return NULL;
    }

    size_t *valid_rows = (size_t *)malloc(MATRIX_ROWS(matrix) * sizeof(size_t));
    if (!valid_rows) {
        printf("Error: Unable to allocate memory for valid rows.\n");
        return NULL;
    }
    size_t valid_count = 0;

    for (size_t i = 0; i < MATRIX_ROWS(matrix); i++) {
        double *row = &matrix->mat[i * MATRIX_COLS(matrix)];
        if (condition(row, MATRIX_COLS(matrix))) {
            valid_rows[valid_count++] = i;
        }
    }

    size_t dims[] = {valid_count, MATRIX_COLS(matrix)};
    simple_matrix *filtered = allocate_matrix(2, dims, ZEROS, 0.0, 0.0);
    if (!filtered) {
        printf("Error: Unable to allocate memory for filtered matrix.\n");
        free(valid_rows);
        return NULL;
    }

    for (size_t i = 0; i < valid_count; i++) {
        size_t row_index = valid_rows[i];
        for (size_t j = 0; j < MATRIX_COLS(matrix); j++) {
            filtered->mat[i * MATRIX_COLS(matrix) + j] = matrix->mat[row_index * MATRIX_COLS(matrix) + j];
        }
    }

    free(valid_rows);

    return filtered;
}





#endif

#endif
