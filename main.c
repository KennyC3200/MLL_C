#include "types.h"
#include "arena.h"
#include "arena.c"
#include "rand.h"
#include "rand.c"

#define MATRIX_SZ(mat) ((u64)((mat->rows) * (mat->cols)))
#define MATRIX_SAME_SZ(a, b) (((a->rows) == (b->rows) && (a->cols) == (b->cols)))

typedef struct {
    u32 rows, cols;

    // Row major
    f32* data;
} matrix;

matrix* matrix_create(mem_arena* arena, u32 rows, u32 cols);
matrix* matrix_load(mem_arena* arena, u32 rows, u32 cols, const char* file);
bool matrix_copy(matrix* dst, matrix* src);
void matrix_clear(matrix* mat);
void matrix_fill(matrix* mat, f32 val);
void matrix_fill_rand(matrix* mat, f32 min, f32 max);

void matrix_scale(matrix* mat, f32 scale);
f32 matrix_sum(matrix* mat);
bool matrix_add(matrix* out, const matrix* a, const matrix* b);
bool matrix_sub(matrix* out, const matrix* a, const matrix* b);
u32 matrix_argmax(matrix* mat);
bool matrix_mul(
    matrix* out, const matrix* a, const matrix* b, 
    bool zero_out, bool transpose_a, bool transpose_b);

// ReLU function (rectified linear unit)
bool matrix_relu(matrix* out, const matrix* in);

// Converts input vector into probability distribution
bool matrix_softmax(matrix* out, const matrix* in);

/* 
Cross-entropy is a measure of the difference between two probability distributions
In ML, it is the standard loss function for classification models
Quantifies performance by penalizing incorrect predictions, with higher penalties for
high-confidence mistakes
p: expected probability distribution
q: actual probability distribution
*/
bool matrix_cross_entropy(matrix* out, const matrix* p, const matrix* q);

bool matrix_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
bool matrix_softmax_add_grad(matrix* out, const matrix* softmax_out, const matrix* grad);
bool matrix_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad);

#define MODEL_VAR_MAX_INPUTS 2
#define MODEL_VAR_NUM_INPUTS(op) \
    ((op) < _MODEL_VAR_OP_UNARY_START ? 0 : ((op) < _MODEL_VAR_OP_BINARY_START ? 1 : 2))

typedef enum {
    MODEL_VAR_FLAG_NONE           = 0,
    MODEL_VAR_FLAG_REQUIRES_GRAD  = (1 << 0),
    MODEL_VAR_FLAG_PARAMETER      = (1 << 1),
    MODEL_VAR_FLAG_INPUT          = (1 << 2),
    MODEL_VAR_FLAG_OUTPUT         = (1 << 3),
    MODEL_VAR_FLAG_DESIRED_OUTPUT = (1 << 4),
    MODEL_VAR_FLAG_COST           = (1 << 5),
} model_flags;

typedef enum {
    MODEL_VAR_OP_NULL = 0,
    MODEL_VAR_OP_CREATE,

    _MODEL_VAR_OP_UNARY_START,
    MODEL_VAR_OP_RELU,
    MODEL_VAR_OP_SOFTMAX,

    _MODEL_VAR_OP_BINARY_START,
    MODEL_VAR_OP_ADD,
    MODEL_VAR_OP_SUB,
    MODEL_VAR_OP_MATMUL,
    MODEL_VAR_OP_CROSS_ENTROPY,
} model_var_op;

typedef struct model {
    u32 idx;
    u32 flags;

    matrix* val;
    matrix* grad;

    model_var_op op;

    struct model* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

typedef struct {
    model_var** vars;
    u32 size;
} model_program;

typedef struct {
    u32 num_vars;

    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;
    model_program cost_prog;
} model_context;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

model_context* model_create(mem_arena* arena);
model_var* model_var_create(
    mem_arena* arena, model_context* model, 
    u32 rows, u32 cols, u32 flags);
model_var* model_var_relu(
    mem_arena* arena, model_context* model, 
    model_var* input, u32 flags);
model_var* model_var_softmax(
    mem_arena* arena, model_context* model, 
    model_var* input, u32 flags);
model_var* model_var_add(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags);
model_var* model_var_sub(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags);
model_var* model_var_matmul(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags);
model_var* model_var_cross_entropy(
    mem_arena* arena, model_context* model, 
    model_var* p, model_var* q, u32 flags);

model_program model_program_create(mem_arena* arena, model_context* model, model_var* out_var);
void model_program_compute(model_program* prog);
void model_program_compute_grads(model_program* prog);

model_context* model_context_create(mem_arena* arena);
void model_compile(mem_arena* arena, model_context* model);
void model_feed_forward(model_context* model);
void model_train(model_context* model, const model_training_desc* training_desc);

void draw_mnist_digit(f32* data);
void create_mnist_model(mem_arena* arena, model_context* model);

void print_matrix(const char* label, matrix* mat) {
    printf("%s:\n", label);
    for (u32 i = 0; i < mat->rows; i++) {
        for (u32 j = 0; j < mat->cols; j++) {
            printf("%.2f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int main(void) {
    mem_arena* arena = arena_create(GiB(1), MiB(1));

    // Each row is one training example
    matrix* train_images = matrix_load(arena, 60000, 28 * 28, "train_images.mat");
    matrix* test_images = matrix_load(arena, 10000, 28 * 28, "test_images.mat");
    matrix* train_labels = matrix_create(arena, 60000, 10);
    matrix* test_labels = matrix_create(arena, 10000, 10);

    // Make the labels matrix data into one-hot encoding
    // By default, the label value is literally the `value`
    {
        matrix* train_labels_file = matrix_load(arena, 60000, 1, "train_labels.mat");
        matrix* test_labels_file = matrix_load(arena, 10000, 1, "test_labels.mat");

        for (u32 i = 0; i < 60000; i++)
            train_labels->data[i * 10 + (u32)train_labels_file->data[i]] = 1.0f;
        for (u32 i = 0; i < 10000; i++)
            test_labels->data[i * 10 + (u32)test_labels_file->data[i]] = 1.0f;
    }

    draw_mnist_digit(train_images->data);
    for (u32 i = 0; i < 10; i++)
        printf("%.0f ", train_labels->data[i]);
    printf("\n\n");

    model_context* model = model_create(arena);
    create_mnist_model(arena, model);
    model_compile(arena, model);

    // Pre-training output
    memcpy(model->input->val->data, train_images->data, sizeof(f32) * 28 * 28);
    model_feed_forward(model);

    printf("Pre-training output: ");
    for (u32 i = 0; i < 10; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }
    printf("\n");

    model_training_desc training_desc = {
        .train_images = train_images,
        .train_labels = train_labels,
        .test_images = test_images,
        .test_labels = test_labels,

        .epochs = 300,
        .batch_size = 50,
        .learning_rate = 0.01f
    };

    model_train(model, &training_desc);

    // Post-training output
    memcpy(model->input->val->data, train_images->data, sizeof(f32) * 28 * 28);
    model_feed_forward(model);

    printf("Post-training output: ");
    for (u32 i = 0; i < 10; i++) {
        printf("%.2f ", model->output->val->data[i]);
    }

    arena_destroy(arena);

    return 0;
}

void draw_mnist_digit(f32* data) {
    // The images are 28 by 28
    for (u32 y = 0; y < 28; y++) {
        for (u32 x = 0; x < 28; x++) {
            f32 num = data[y * 28 + x];
            u32 color = 232 + (u32)(num * 24); // [232, 256]
            printf("\x1b[48;5;%dm   ", color);
        }
        printf("\n");
    }
    printf("\x1b[0m");
}

void create_mnist_model(mem_arena* arena, model_context* model) {
    // Input layer
    model_var* input = model_var_create(arena, model, 28 * 28, 1, MODEL_VAR_FLAG_INPUT);

    const u32 flags = MODEL_VAR_FLAG_REQUIRES_GRAD | MODEL_VAR_FLAG_PARAMETER;

    model_var* W0 = model_var_create(arena, model, 16, 28 * 28, flags);
    model_var* W1 = model_var_create(arena, model, 16, 16, flags);
    model_var* W2 = model_var_create(arena, model, 10, 16, flags); // output has 10 variables

    // Xavier intialization
    f32 bound0 = sqrtf(6.0f / (28 * 28 + 16));
    f32 bound1 = sqrtf(6.0f / (16 + 16));
    f32 bound2 = sqrtf(6.0f / (16 + 10));
    matrix_fill_rand(W0->val, -bound0, bound0);
    matrix_fill_rand(W1->val, -bound1, bound1);
    matrix_fill_rand(W2->val, -bound2, bound2);

    model_var* b0 = model_var_create(arena, model, 16, 1, flags);
    model_var* b1 = model_var_create(arena, model, 16, 1, flags);
    model_var* b2 = model_var_create(arena, model, 10, 1, flags);
    
    // First layer 
    model_var* z0_a = model_var_matmul(arena, model, W0, input, 0);
    model_var* z0_b = model_var_add(arena, model, z0_a, b0, 0);
    model_var* a0 = model_var_relu(arena, model, z0_b, 0);

    // Second layer
    model_var* z1_a = model_var_matmul(arena, model, W1, a0, 0);
    model_var* z1_b = model_var_add(arena, model, z1_a, b1, 0);
    model_var* z1_c = model_var_relu(arena, model, z1_b, 0);
    model_var* a1 = model_var_add(arena, model, z1_c, a0, 0); // Residual connection

    // Output layer
    model_var* z2_a = model_var_matmul(arena, model, W2, a1, 0);
    model_var* z2_b = model_var_add(arena, model, z2_a, b2, 0);
    model_var* output = model_var_softmax(arena, model, z2_b, MODEL_VAR_FLAG_OUTPUT);

    model_var* desired_output = model_var_create(
        arena, model, 10, 1, MODEL_VAR_FLAG_DESIRED_OUTPUT);

    model_var* cost = model_var_cross_entropy(
        arena, model, desired_output, output, MODEL_VAR_FLAG_COST);
}

matrix* matrix_create(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);

    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, (u64)(rows * cols));

    return mat;
}

matrix* matrix_load(mem_arena* arena, u32 rows, u32 cols, const char* file) {
    matrix* mat = matrix_create(arena, rows, cols);

    FILE* f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    size = MIN(size, sizeof(f32) * rows * cols);

    fread(mat->data, 1, size, f);

    return mat;
}

bool matrix_copy(matrix* dst, matrix* src) {
    if (!MATRIX_SAME_SZ(dst, src)) return false;

    memcpy(dst->data, src->data, sizeof(f32) * (u64)(dst->rows * dst->cols));
    return true;
}

void matrix_clear(matrix* mat) {
    memset(mat->data, 0, sizeof(f32) * (u64)(mat->rows * mat->cols));
}

void matrix_fill(matrix* mat, f32 val) {
    u64 size = MATRIX_SZ(mat);

    for (u64 i = 0; i < size; i++)
        mat->data[i] = val;
}

void matrix_fill_rand(matrix* mat, f32 min, f32 max) {
    u64 size = MATRIX_SZ(mat);

    for (u64 i = 0; i < size; i++)
        mat->data[i] = prng_randf() * (max - min) + min;
}

void matrix_scale(matrix* mat, f32 scale) {
    u64 size = MATRIX_SZ(mat);

    for (u64 i = 0; i < size; i++)
        mat->data[i] *= scale;
}

f32 matrix_sum(matrix* mat) {
    u64 size = MATRIX_SZ(mat);
    f32 sum = 0.0f;

    for (u64 i = 0; i < size; i++)
        sum += mat->data[i];
    
    return sum;
}

bool matrix_add(matrix* out, const matrix* a, const matrix* b) {
    if (!MATRIX_SAME_SZ(a, b) || !MATRIX_SAME_SZ(out, a)) return false;

    u64 size = MATRIX_SZ(out);
    for (u64 i = 0; i < size; i++)
        out->data[i] = a->data[i] + b->data[i];
    
    return true;
}

bool matrix_sub(matrix* out, const matrix* a, const matrix* b) {
    if (!MATRIX_SAME_SZ(a, b) || !MATRIX_SAME_SZ(out, a)) return false;

    u64 size = MATRIX_SZ(out);
    for (u64 i = 0; i < size; i++)
        out->data[i] = a->data[i] - b->data[i];
    
    return true;
}

u32 matrix_argmax(matrix* mat) {
    u64 size = MATRIX_SZ(mat);

    u32 argmax = 0;
    for (u64 i = 0; i < size; i++) {
        if (mat->data[i] > mat->data[argmax]) argmax = i;
    }

    return argmax;
}

// The two characters in the suffix can be N (not transpose) or T (transpose)
void _matrix_mul_NN(matrix* out, const matrix* a, const matrix* b) {
    // Optimized for cache: i -> k -> j
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 k = 0; k < a->cols; k++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[i * out->cols + j] +=
                    a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
}

void _matrix_mul_NT(matrix* out, const matrix* a, const matrix* b) {
    // Optimized for cache: i -> j -> k
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->cols; k++) {
                out->data[i * out->cols + j] +=
                    a->data[i * a->cols + k] * b->data[j * b->cols + k];
            }
        }
    }
}

void _matrix_mul_TN(matrix* out, const matrix* a, const matrix* b) {
    // Optimized for cache: k -> i -> j
    for (u64 k = 0; k < a->rows; k++) {
        for (u64 i = 0; i < out->rows; i++) {
            for (u64 j = 0; j < out->cols; j++) {
                out->data[i * out->cols + j] +=
                    a->data[k * a->cols + i] * b->data[k * b->cols + j];
            }
        }
    }
}

void _matrix_mul_TT(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            for (u64 k = 0; k < a->rows; k++) {
                out->data[i * out->cols + j] +=
                    a->data[k * a->cols + i] * b->data[j * b->cols + k];
            }
        }
    }
}

bool matrix_mul(
    matrix* out, const matrix* a, const matrix* b, 
    bool zero_out, bool transpose_a, bool transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows,
        a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows,
        b_cols = transpose_b ? b->rows : b->cols;
    
    if (a_cols != b_rows) return false;
    if (out->rows != a_rows || out->cols != b_cols) return false;

    if (zero_out) matrix_clear(out);

    u32 transpose = (transpose_a << 1) + transpose_b;
    switch (transpose) {
        case 0b00: { _matrix_mul_NN(out, a, b); } break;
        case 0b01: { _matrix_mul_NT(out, a, b); } break;
        case 0b10: { _matrix_mul_TN(out, a, b); } break;
        case 0b11: { _matrix_mul_TT(out, a, b); } break;
        default: break;
    }

    return true;
}

bool matrix_relu(matrix* out, const matrix* in) {
    if (!MATRIX_SAME_SZ(in, out)) return false;

    u64 size = MATRIX_SZ(out);
    for (u64 i = 0; i < size; i++)
        out->data[i] = MAX(0, in->data[i]);

    return true;
}

bool matrix_softmax(matrix* out, const matrix* in) {
    if (!MATRIX_SAME_SZ(in, out)) return false;

    u64 size = MATRIX_SZ(out);
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i]);
        sum += out->data[i];
    }

    matrix_scale(out, 1.0f / sum);

    return true;
}

bool matrix_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    if (!MATRIX_SAME_SZ(p, q) || !MATRIX_SAME_SZ(p, out)) return false;

    u64 size = MATRIX_SZ(out);
    for (u64 i = 0; i < size; i++)
        out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q->data[i]);

    return true;
}

bool matrix_relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    if (!MATRIX_SAME_SZ(in, out)) return false;
    if (!MATRIX_SAME_SZ(out, grad)) return false;

    u64 size = MATRIX_SZ(out);
    for (u64 i = 0; i < size; i++)
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;

    return true;
}

bool matrix_softmax_add_grad(matrix* out, const matrix* softmax_out, const matrix* grad) {
    // This is in the final step of the neural network, so the output should be a vector
    if (softmax_out->rows != 1 && softmax_out->cols != 1) return false;

    // We can also calculate the gradient via the Jacobian-vector product (I believe), since the 
    // loss function outputs a singular scalar value
    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    // Safeguard, because this Jacobian is a square matrix
    u32 size = MAX(softmax_out->rows, softmax_out->cols);
    matrix* jacobian = matrix_create(scratch.arena, size, size);

    // Calculate Jacobian matrix values
    for (u32 i = 0; i < size; i++) {
        for (u32 j = 0; j < size; j++) {
            jacobian->data[i * size + j] = 
                softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
        }
    }

    matrix_mul(out, jacobian, grad, 0, 0, 0);

    arena_scratch_release(scratch);

    return true;
}

bool matrix_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad) 
{
    if (!MATRIX_SAME_SZ(p, q))
        return false;

    u64 size = MATRIX_SZ(p);

    if (p_grad != NULL) {
        if (!MATRIX_SAME_SZ(p, p_grad))
            return false;

        for (u64 i = 0; i < size; i++)
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
    }

    if (q_grad != NULL) {
        if (!MATRIX_SAME_SZ(q, q_grad))
            return false;

        for (u64 i = 0; i < size; i++)
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
    }

    return true;
}

model_context* model_create(mem_arena* arena) {
    model_context* model = PUSH_STRUCT(arena, model_context);
    return model;
}

model_var* _model_var_unary_impl(
    mem_arena* arena, model_context* model,
    model_var* input, u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    // This is due to chain rule requiring a gradient value
    if (input->flags & MODEL_VAR_FLAG_REQUIRES_GRAD)
        flags |= MODEL_VAR_FLAG_REQUIRES_GRAD;

    model_var* out = model_var_create(arena, model, rows, cols, flags);
    out->op = op;
    out->inputs[0] = input;

    return out;
}

model_var* _model_var_binary_impl(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b,
    u32 rows, u32 cols,
    u32 flags, model_var_op op
) {
    if ((a->flags | b->flags) & MODEL_VAR_FLAG_REQUIRES_GRAD)
        flags |= MODEL_VAR_FLAG_REQUIRES_GRAD;
    
    model_var* out = model_var_create(arena, model, rows, cols, flags);
    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;

    return out;
}

model_var* model_var_create(
    mem_arena* arena, model_context* model, 
    u32 rows, u32 cols, u32 flags
) {
    model_var* out = PUSH_STRUCT(arena, model_var);

    out->idx = model->num_vars++;
    out->val = matrix_create(arena, rows, cols);
    out->flags = flags;
    out->op = MODEL_VAR_OP_CREATE;

    // Some variables don't have gradients--they are constants
    if (flags & MODEL_VAR_FLAG_REQUIRES_GRAD)  
        out->grad = matrix_create(arena, rows, cols);

    // If model variable is used as input/output/desired output/cost variables
    if (flags & MODEL_VAR_FLAG_INPUT)          model->input = out;
    if (flags & MODEL_VAR_FLAG_OUTPUT)         model->output = out;
    if (flags & MODEL_VAR_FLAG_DESIRED_OUTPUT) model->desired_output = out;
    if (flags & MODEL_VAR_FLAG_COST)           model->cost = out;

    return out;
}

model_var* model_var_relu(
    mem_arena* arena, model_context* model, 
    model_var* input, u32 flags
) {
    return _model_var_unary_impl(
        arena, model, 
        input, input->val->rows, input->val->cols, 
        flags, MODEL_VAR_OP_RELU);
}

model_var* model_var_softmax(
    mem_arena* arena, model_context* model, 
    model_var* input, u32 flags
) {
    return _model_var_unary_impl(
        arena, model,
        input, input->val->rows, input->val->cols,
        flags, MODEL_VAR_OP_SOFTMAX);
}

model_var* model_var_add(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
) {
    if (!MATRIX_SAME_SZ(a->val, b->val))
        return NULL;

    return _model_var_binary_impl(
        arena, model, 
        a, b, 
        a->val->rows, a->val->cols, 
        flags, MODEL_VAR_OP_ADD);
}

model_var* model_var_sub(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
) {
    if (!MATRIX_SAME_SZ(a->val, b->val))
        return NULL;

    return _model_var_binary_impl(
        arena, model, 
        a, b, 
        a->val->rows, a->val->cols, 
        flags, MODEL_VAR_OP_SUB);
}

model_var* model_var_matmul(
    mem_arena* arena, model_context* model, 
    model_var* a, model_var* b, u32 flags
) {
    if (a->val->cols != b->val->rows)
        return NULL;

    return _model_var_binary_impl(
        arena, model, 
        a, b, 
        a->val->rows, b->val->cols,
        flags, MODEL_VAR_OP_MATMUL);
}

model_var* model_var_cross_entropy(
    mem_arena* arena, model_context* model, 
    model_var* p, model_var* q, u32 flags
) {
    if (!MATRIX_SAME_SZ(p->val, q->val))
        return NULL;
    
    return _model_var_binary_impl(
        arena, model,
        p, q,
        p->val->rows, p->val->cols,
        flags, MODEL_VAR_OP_CROSS_ENTROPY);
}

model_program model_program_create(mem_arena* arena, model_context* model, model_var* out_var) {
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);

    bool* visited = PUSH_ARRAY(scratch.arena, bool, model->num_vars);

    u32 stack_size = 0;
    u32 out_size = 0;
    // TODO: stack has fixed size of num_vars, which could cause issues
    model_var** stack = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);
    model_var** out = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars);

    stack[stack_size++] = out_var;
    while (stack_size > 0) {
        model_var* curr = stack[--stack_size];

        // Check for index out of bounds
        if (curr->idx >= model->num_vars) continue;

        if (visited[curr->idx]) {
            if (out_size < model->num_vars) out[out_size++] = curr;
        } else {
            visited[curr->idx] = true;

            if (stack_size < model->num_vars) stack[stack_size++] = curr;

            for (u32 i = 0, n = MODEL_VAR_NUM_INPUTS(curr->op); i < n; i++) {
                model_var* input = curr->inputs[i];

                if (input->idx >= model->num_vars || visited[input->idx]) continue;

                // If `input` is in the stack, move it to the top of the stack
                for (u32 j = 0; j < stack_size; j++) {
                    if (stack[j] == input) {
                        for (u32 k = j; k < stack_size - 1; k++) {
                            stack[k] = stack[k + 1];
                        }
                        stack_size--;
                    }
                }

                if (stack_size < model->num_vars) stack[stack_size++] = input;
            }
        }
    }

    model_program prog = {
        .size = out_size,
        .vars = PUSH_ARRAY_NZ(arena, model_var*, out_size)
    };
    memcpy(prog.vars, out, out_size * sizeof(model_var*));

    arena_scratch_release(scratch);

    return prog;
}

void model_program_compute(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* curr = prog->vars[i];
        model_var* a = curr->inputs[0];
        model_var* b = curr->inputs[1];

        switch (curr->op) {
            case MODEL_VAR_OP_NULL:
            case MODEL_VAR_OP_CREATE: break;

            case _MODEL_VAR_OP_UNARY_START: break;
            case MODEL_VAR_OP_RELU: { matrix_relu(curr->val, a->val); } break;
            case MODEL_VAR_OP_SOFTMAX: { matrix_softmax(curr->val, a->val); } break;

            case _MODEL_VAR_OP_BINARY_START: break;
            case MODEL_VAR_OP_ADD: { matrix_add(curr->val, a->val, b->val); } break;
            case MODEL_VAR_OP_SUB: { matrix_sub(curr->val, a->val, b->val); } break;
            case MODEL_VAR_OP_MATMUL: { 
                matrix_mul(curr->val, a->val, b->val, true, false, false); 
            } break;
            case MODEL_VAR_OP_CROSS_ENTROPY: {
                matrix_cross_entropy(curr->val, a->val, b->val);
            } break;
        }
    }
}

void model_program_compute_grads(model_program* prog) {
    // Only keep gradient if it requires the gradient and is not a parameter
    for (u32 i = 0; i < prog->size; i++) {
        model_var* curr = prog->vars[i];

        if ((curr->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) != MODEL_VAR_FLAG_REQUIRES_GRAD) continue;
        if (curr->flags & MODEL_VAR_FLAG_PARAMETER) continue;

        matrix_clear(curr->grad);
    }

    // The gradient WRT the final variable in the program is 1
    // In the future, make this explicitly a variable that is a scalar
    matrix_fill(prog->vars[prog->size - 1]->grad, 1.0f);

    for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
        model_var* curr = prog->vars[i];

        if (!(curr->flags & MODEL_VAR_FLAG_REQUIRES_GRAD)) continue;

        model_var* a = curr->inputs[0];
        model_var* b = curr->inputs[1];

        u32 num_inputs = MODEL_VAR_NUM_INPUTS(curr->op);

        if (
            num_inputs == 1 && 
            (a->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) != MODEL_VAR_FLAG_REQUIRES_GRAD
        ) continue;

        if (
            num_inputs == 2 &&
            (a->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) != MODEL_VAR_FLAG_REQUIRES_GRAD &&
            (b->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) != MODEL_VAR_FLAG_REQUIRES_GRAD
        ) continue;

        switch (curr->op) {
            case MODEL_VAR_OP_NULL:
            case MODEL_VAR_OP_CREATE: break;

            case _MODEL_VAR_OP_UNARY_START: break;
            case MODEL_VAR_OP_RELU: {
                matrix_relu_add_grad(a->grad, a->val, curr->grad);
            } break;
            case MODEL_VAR_OP_SOFTMAX: {
                matrix_softmax_add_grad(a->grad, curr->val, curr->grad);
            }; break;

            case _MODEL_VAR_OP_BINARY_START: break;
            case MODEL_VAR_OP_ADD: {
                if (a->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) 
                    matrix_add(a->grad, a->grad, curr->grad);
                if (b->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) 
                    matrix_add(b->grad, b->grad, curr->grad);
            } break;
            case MODEL_VAR_OP_SUB: {
                if (a->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) 
                    matrix_add(a->grad, a->grad, curr->grad);
                if (b->flags & MODEL_VAR_FLAG_REQUIRES_GRAD) 
                    matrix_sub(b->grad, b->grad, curr->grad);
            } break;
            case MODEL_VAR_OP_MATMUL: {
                if (a->flags & MODEL_VAR_FLAG_REQUIRES_GRAD)
                    matrix_mul(a->grad, curr->grad, b->val, 0, 0, 1);
                if (b->flags & MODEL_VAR_FLAG_REQUIRES_GRAD)
                    matrix_mul(b->grad, a->val, curr->grad, 0, 1, 0);
            } break;
            case MODEL_VAR_OP_CROSS_ENTROPY: {
                matrix_cross_entropy_add_grad(a->grad, b->grad, a->val, b->val, curr->grad);
            } break;
        }
    }
}

model_context* model_context_create(mem_arena* arena) {
    model_context* model = PUSH_STRUCT(arena, model_context);
    return model;
}

void model_compile(mem_arena* arena, model_context* model) {
    if (model->output != NULL)
        model->forward_prog = model_program_create(arena, model, model->output);
    
    if (model->cost != NULL)
        model->cost_prog = model_program_create(arena, model, model->cost);
}

void model_feed_forward(model_context* model) {
    model_program_compute(&model->forward_prog);
}

void model_train(model_context* model, const model_training_desc* training_desc) {
    matrix* train_images = training_desc->train_images;
    matrix* train_labels = training_desc->train_labels;
    matrix* test_images = training_desc->test_images;
    matrix* test_labels = training_desc->test_labels;

    u32 num_examples = train_images->rows;
    u32 num_tests = test_images->rows;
    u32 input_size = train_images->cols;
    u32 output_size = train_labels->cols;

    u32 num_batches = num_examples / training_desc->batch_size;

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);

    u32* training_order = PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
    for (u32 i = 0; i < num_examples; i++) training_order[i] = i;

    for (u32 epoch = 0; epoch < training_desc->epochs; epoch++) {
        // Mix up the training order
        for (u32 i = 0; i < num_examples; i++) {
            // prng_rand() % num_examples is technically not uniform unless num_examples is a power
            // of 2, but its fine
            u32 a = prng_rand() % num_examples;
            u32 b = prng_rand() % num_examples;

            u32 tmp = training_order[b];
            training_order[b] = training_order[a];
            training_order[a] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            // Will be using stochastic gradient descent--more computationally efficient
            // This involves clearing the gradients of the parameters for every batch
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var* curr = model->cost_prog.vars[i];
                if (curr->flags & MODEL_VAR_FLAG_PARAMETER) matrix_clear(curr->grad);
            }

            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < training_desc->batch_size; i++) {
                u32 order_idx = batch * training_desc->batch_size + i;
                u32 idx = training_order[order_idx];

                // Copy the input data to the network
                memcpy(
                    model->input->val->data,
                    train_images->data + idx * input_size,
                    sizeof(f32) * input_size);

                // Copy the desired output data into the model
                memcpy(
                    model->desired_output->val->data,
                    train_labels->data + idx * output_size,
                    sizeof(f32) * output_size);

                // Compute the cost and gradients for the variables in the network
                model_program_compute(&model->cost_prog);       // forward pass
                model_program_compute_grads(&model->cost_prog); // backprop

                avg_cost += matrix_sum(model->cost->val);
            }

            avg_cost /= (f32)training_desc->batch_size;

            for (u32 i = 0; i < model->cost_prog.size; i++) {
                model_var* curr = model->cost_prog.vars[i];
                if ((curr->flags & MODEL_VAR_FLAG_PARAMETER) != MODEL_VAR_FLAG_PARAMETER) continue;

                // Learning rate is something to make gradients more tame and obviously we divide
                // by the batch size to get the average gradient
                matrix_scale(curr->grad, training_desc->learning_rate / training_desc->batch_size);

                // Subtract because we want gradient descent--go in the opposite direction of
                // the gradient
                matrix_sub(curr->val, curr->val, curr->grad);
            }

            printf(
                "Epoch %2d / %2d, Batch %4d / %4d, Average Cost: %.4f\r",
                epoch + 1, training_desc->epochs,
                batch + 1, num_batches, avg_cost
            );
            fflush(stdout);
        }
        printf("\n");

        u32 num_correct = 0;
        f32 avg_cost = 0;
        for (u32 i = 0; i < num_tests; i++) {
            // Send the input data into the model
            memcpy(
                model->input->val->data,
                test_images->data + i * input_size,
                sizeof(f32) * input_size);

            // Copy the expected output into the model
            memcpy(
                model->desired_output->val->data,
                test_labels->data + i * output_size,
                sizeof(f32) * output_size);
            
            model_program_compute(&model->cost_prog);

            avg_cost += matrix_sum(model->cost->val);
            num_correct += 
                matrix_argmax(model->desired_output->val) == matrix_argmax(model->output->val);
        }

        avg_cost /= (f32) num_tests;
        printf(
            "Test Completed. Accuracy: %5d / %5d (%.1f%%), Average Cost: %.4f\n",
            num_correct, num_tests, (f32)num_correct / num_tests * 100.0f, avg_cost);
    }

    arena_scratch_release(scratch);
}