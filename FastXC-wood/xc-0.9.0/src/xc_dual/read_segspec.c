#include "read_segspec.h"
#include <dirent.h>
#include <sys/stat.h>  // Include for stat()

/* read the segment spectrum and return whole spec array
 * and header using preallocated buffer
 *
 * wuchao@20211004
 * */
complex *read_spec_buffer(char *name, SEGSPEC *hd, complex *buffer)
{
  FILE *strm = NULL;
  int size;

  if ((strm = fopen(name, "rb")) == NULL)
  {
    fprintf(stderr, "Unable to open %s\n", name);
    return NULL;
  }

  if (fread(hd, sizeof(SEGSPEC), 1, strm) != 1)
  {
    fprintf(stderr, "Error in reading SEGSPEC header %s\n", name);
    return NULL;
  }

  /* read whole segment spectrum in
   * Total size is nseg*nspec*sizeof(our_float_complex) */
//   size = sizeof(complex) * hd->nspec * hd->nstep;

  
//   if (fread((char *)buffer, size, 1, strm) != 1)
//   {
//     fprintf(stderr, "Error in reading SEGSPEC data %s\n", name);
//     return NULL;
//   }
  

  fclose(strm);

  return buffer;
}

/* read the segment spectrum header
 *
 * wuchao@20211004
 * */

int read_spechead(const char *name, SEGSPEC *hd)
{
  FILE *strm;

  if ((strm = fopen(name, "rb")) == NULL)
  {
    fprintf(stderr, "Unable to open %s\n", name);
    return -1;
  }

  if (fread(hd, sizeof(SEGSPEC), 1, strm) != 1)
  {
    fprintf(stderr, "Error in reading SAC header %s\n", name);
    fclose(strm);
    return -1;
  }

  fclose(strm);
  return 0;
}

// ----------------------------------------- Read Merge Segspec Files -----------------------------------------
void readMergedSpecBuffer(const char *filename, complex *spec_buffer, size_t buffer_size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open %s\n", filename);
        return;
    }

    // 确定文件的总大小
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    long dataEndPosition = fileSize - sizeof(int);  // 不包括文件末尾的整型文件数
    rewind(file);

    SEGSPEC hd;
    size_t current_offset = 0;  // 当前在spec_buffer中的偏移量（以complex为单位计数）

    while (ftell(file) < dataEndPosition && fread(&hd, sizeof(SEGSPEC), 1, file) == 1) {
        size_t num_complex = hd.nspec * hd.nstep;  // 计算当前segment包含的complex数量

        if (current_offset + num_complex > buffer_size) {
            fprintf(stderr, "Buffer overflow error: not enough space in spec_buffer\n");
            fclose(file);
            return;  // Buffer overflow
        }

        // 直接读取到传入的spec_buffer中的适当位置
        if (fread(spec_buffer + current_offset, sizeof(complex), num_complex, file) != num_complex) {
            fprintf(stderr, "Error reading SEGSPEC data from %s\n", filename);
            fclose(file);
            return;
        }

        current_offset += num_complex;  // 更新缓冲区偏移量（以complex为单位）
    }

    // printf("Read total %zu complex values.\n", current_offset);

    fclose(file);
}

void readMergedSpecBuffer_onlyOneHead(const char *filename, complex *spec_buffer, size_t buffer_size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open %s\n", filename);
        return;
    }

    // 首先读取并忽略头区
    SEGSPEC hd;
    if (fread(&hd, sizeof(SEGSPEC), 1, file) != 1) {
        fprintf(stderr, "Error reading SEGSPEC header from %s\n", filename);
        fclose(file);
        return;
    }

    // 计算数据部分的总大小（忽略最后的整数大小）
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    long dataSectionSize = fileSize - sizeof(int) - sizeof(SEGSPEC); // 减去头区和最后的整型文件数大小

    // 重定位到数据部分开始的位置
    rewind(file);
    fseek(file, sizeof(SEGSPEC), SEEK_SET);

    if (dataSectionSize > buffer_size) {
        fprintf(stderr, "Buffer overflow error: not enough space in spec_buffer\n");
        fclose(file);
        return;
    }

    if (fread(spec_buffer, dataSectionSize, 1, file) != 1) {
        fprintf(stderr, "Error reading all data from %s\n", filename);
        fclose(file);
        return;
    }

    int fileCount;
    fseek(file, -sizeof(int), SEEK_END);
    if (fread(&fileCount, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Failed to read the number of merged files from %s\n", filename);
    } else {
        printf("Number of files merged: %d\n", fileCount);
    }

    printf("Total %zu complex values read from the data section.\n", dataSectionSize / sizeof(complex));

    fclose(file);
}

int readMergedSpec_head(const char *filename, SEGSPEC *spec_head) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Unable to open %s\n", filename);
        return -1;
    }

    // 移动到文件末尾前的int大小位置
    fseek(file, -sizeof(int), SEEK_END);
    int fileCount;
    fread(&fileCount, sizeof(int), 1, file);

    // 移动回文件开始读取SEGSPEC头信息
    rewind(file);

    // Now read the SEGSPEC header
    if (fread(spec_head, sizeof(SEGSPEC), 1, file) != 1) {
        fprintf(stderr, "Error reading SEGSPEC header from %s\n", filename);
        fclose(file);
        return -1;
    }
    fclose(file);
    return fileCount;
}

int mergeFiles(const char *baseDir, FILE *outputFile) {
    DIR *dir;
    struct dirent *entry;
    char path[1024];
    int file_num = 0;

    // 打开目录
    dir = opendir(baseDir);
    if (dir == NULL) {
        perror("Directory not found");
        return -1;
    }

    // 遍历目录
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;  // 忽略隐藏文件和目录
        }

        snprintf(path, sizeof(path), "%s/%s", baseDir, entry->d_name);
        
        // 使用 stat() 来确定是目录还是文件
        struct stat statbuf;
        if (stat(path, &statbuf) == 0) {
            if (S_ISDIR(statbuf.st_mode)) {
                // 如果是目录，递归调用处理子目录
                file_num += mergeFiles(path, outputFile);
            } else {
                // 检查文件扩展名
                if (strstr(entry->d_name, ".segspec") != NULL) {
                    printf("Processing file: %s\n", path);  // 打印路径和文件名
                    FILE *file = fopen(path, "rb");
                    if (file) {
                        char buffer[4096];
                        size_t bytes;
                        size_t totalBytes = 0;  // 用于累计每个文件的写入字节数
                        while ((bytes = fread(buffer, 1, sizeof(buffer), file)) > 0) {
                            fwrite(buffer, 1, bytes, outputFile);
                            totalBytes += bytes;  // 累加写入的字节数
                        }
                        float mbWritten = totalBytes / (1024.0 * 1024.0);  // 将字节数转换为MB
                        printf("Written %.2f MB to file\n", mbWritten);
                        fclose(file);
                        file_num++;
                    } else {
                        perror("Failed to open a .segspec file");
                    }
                }
            }
        } else {
            perror("Failed to get file status");
        }
    }
    closedir(dir);
    return file_num;
}

int mergeFiles_onlyOneHead(const char *baseDir, FILE *outputFile, int *firstFile) {
    DIR *dir;
    struct dirent *entry;
    char path[1024];
    int file_num = 0;
    SEGSPEC hd;

    dir = opendir(baseDir);
    if (dir == NULL) {
        perror("Directory not found");
        return -1;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.') continue; // 忽略隐藏文件和目录

        snprintf(path, sizeof(path), "%s/%s", baseDir, entry->d_name);
        struct stat statbuf;
        if (stat(path, &statbuf) != 0) {
            perror("Failed to get file status");
            continue;
        }

        if (S_ISDIR(statbuf.st_mode)) {
            file_num += mergeFiles_onlyOneHead(path, outputFile, firstFile); // 递归处理子目录
        } else if (strstr(entry->d_name, ".segspec") != NULL) {
            printf("Processing file: %s\n", path);
            FILE *file = fopen(path, "rb");
            if (file) {
                if (*firstFile) {
                    // 如果是第一个文件，读取并写入头部
                    if (fread(&hd, sizeof(SEGSPEC), 1, file) == 1) {
                        fwrite(&hd, sizeof(SEGSPEC), 1, outputFile);
                    }
                    *firstFile = 0; // 标记头部已写入，后续文件不再写入头部
                } else {
                    // 跳过头部
                    fseek(file, sizeof(SEGSPEC), SEEK_SET);
                }

                // 读取并写入数据区
                char buffer[4096];
                size_t bytes;
                size_t totalBytes = 0;
                while ((bytes = fread(buffer, 1, sizeof(buffer), file)) > 0) {
                    fwrite(buffer, 1, bytes, outputFile);
                    totalBytes += bytes;
                }
                float mbWritten = totalBytes / (1024.0 * 1024.0);
                printf("Written %.2f MB to file\n", mbWritten);
                fclose(file);
                file_num++;
            } else {
                perror("Failed to open a .segspec file");
            }
        }
    }
    closedir(dir);
    return file_num;
}

