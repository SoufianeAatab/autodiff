#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 65432

typedef float f32;
typedef unsigned int u32;

f32 *get_data(u32 *size) {
    int server_fd;
    int new_socket;
    struct sockaddr_in address;
    socklen_t addrlen;
    int opt = 1;
    addrlen = sizeof(address);
    
    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // Forcefully attaching socket to the port
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Forcefully attaching socket to the port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, &addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    // receive 4 bytes for length indicator
    int bytes_length_count = 0;
    int bytes_length_total = 0;

    u32 length_descriptor = 0;
    char *len_buffer = (char *)&length_descriptor;

    while (bytes_length_total < 4) {
        bytes_length_count = recv(new_socket, &len_buffer[bytes_length_total],
                                  sizeof(u32) - bytes_length_total, 0);

        if (bytes_length_count == -1) {
            perror("recv");
        } else if (bytes_length_count == 0) {
            printf("Unexpected end of transmission.\n");
            close(server_fd);
            exit(EXIT_SUCCESS);
        }

        bytes_length_total += bytes_length_count;
    }

    // receive payload
    int bytes_payload_count = 0;
    int bytes_payload_total = 0;

    size_t data_size = length_descriptor * sizeof(f32);
    f32 *data = (f32 *)malloc(data_size);
    char *buffer = (char *)data;

    while (bytes_payload_total < (int)data_size) {
        bytes_payload_count = recv(new_socket, &buffer[bytes_payload_total],
                                   data_size - bytes_payload_total, 0);

        if (bytes_payload_count == -1) {
            perror("recv");
        } else if (bytes_payload_count == 0) {
            printf("Unexpected end of transmission.\n");
            close(server_fd);
            exit(EXIT_SUCCESS);
        }
        bytes_payload_total += bytes_payload_count;
    }

    *size = length_descriptor;
    close(server_fd);
    return data;
}
