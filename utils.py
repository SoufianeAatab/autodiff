def generate_c_test(id, ground_var, size):
    return f'''
    printf("============================\\nTEST::{ground_var}\\n");
    if(check(&buf[{id}], {ground_var}, {size}))''' + '''{
        printf("\\033[32mOK!\\033[0m\\n");
    } else {
        printf("\\033[31mNO OK!\\033[0m\\n");'''+f'''    
        printf("OUT:\\n");print(&buf[{id}], {size});printf("TRUE:\\n");print({ground_var}, {size});''' +    " }" 

def generate_c_buff(var_name, out):
    output_size = len(out.reshape(-1))
    code = ""
    code += f"float {var_name}[{output_size}] = "
    code += "{"
    for elem in out.reshape(-1):
        code += f"{elem}, "
    # code += elems
    code += "};"
    return code

def generate_c_file(filename, body):
    code = '''
    #include <stdlib.h>
    #include <stdio.h>
    #include <time.h>
    #include "conv2d_cmsis.cc"
    bool check(float* a, float* b, int size){
        for(int i=0;i<size;++i){
            if (fabs(a[i] - b[i]) > 0.00001) return false;
        }
        return true;
    }

    void print(float* a, int size){
        for(int i=0;i<size;++i){
            printf("%f, ", a[i]);
        }
        printf("\\n");
    }
    '''
    code += """
    int main(){
    """ + str(body) + """    
        return 0;
    }
    """
    path = 'code/' + filename + '.c'
    with open(path, 'w') as f:
        f.write(code)

    print(path, 'file generated')
