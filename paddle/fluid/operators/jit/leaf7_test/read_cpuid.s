.text

# rdi: uint32_t leaf
# rsi: uint32_t subleaf
# rdx: uint32_t regs[4]
.globl read_cpuid
read_cpuid:
    push %rbp
    mov %rdi, %rax
    mov %rsi, %rcx
    mov %rdx, %r10
    cpuid
    mov %eax, 0x0(%r10)
    mov %ebx, 0x4(%r10)
    mov %ecx, 0x8(%r10)
    mov %edx, 0xc(%r10)
    pop %rbp
    ret
.type read_cpuid,@function
.size read_cpuid,.-read_cpuid