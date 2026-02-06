def generate_c_test(id, ground_var, size):
    return f'''printf("============================\\nTEST::{ground_var}\\n");
if(check(&buf[{id}], {ground_var}, {size})) {{
    printf("\\033[32mOK!\\033[0m\\n");
}} else {{
    printf("\\033[31mNO OK!\\033[0m\\n");
    printf("OUT:\\n");
    print(&buf[{id}], {size});
    printf("TRUE:\\n");
    print({ground_var}, {size});
}}
'''

def generate_c_buff(var_name, out):
    output_size = len(out.reshape(-1))
    elements = ", ".join(f"{elem}" for elem in out.reshape(-1))
    code = f"float {var_name}[{output_size}] = {{{elements}}};\n"
    return code

def gen_training_loop(code, interpreter, epochs, training_size):
    indented_code = "\n".join("    " + line if line.strip() else "" for line in code.strip().split("\n"))
    indented_temp_assign = "\n".join(line for line in interpreter.temp_assign)
    
    return f'''{interpreter.gen_init_params()}
{indented_temp_assign}
size_t epochs = {epochs};
for (size_t i = 0; i < epochs; ++i) {{
    float loss = 0;
    for (size_t j = 0; j < {training_size}; ++j) {{
{indented_code}
    }}
    printf("Loss: %f\\n", loss / {training_size}.0f);
}}
'''

def generate_c_file(filename, body):
    code = '''#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "conv2d_cmsis.cc"

bool check(float* a, float* b, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > 0.001) return false;
    }
    return true;
}

void print(float* a, int size) {
    for (int i = 0; i < size; ++i) {
        printf("%f, ", a[i]);
    }
    printf("\\n");
}

int main() {
'''
    code += body
    code += '''
    return 0;
}
'''
    path = 'code/' + filename + '.c'
    with open(path, 'w') as f:
        f.write(code)
    
    print(path, 'file generated')
