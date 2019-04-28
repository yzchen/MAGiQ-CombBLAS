## Limitations of current implementation

#### Issue with scaling to very large queries

If you have very large queries those are related to too many entities in the RDF graph,
this means in result generation phase you will have need to keep many columns in one (database) table.

As currently the implementation is hard-coded with maximum 5 columns, 
so it's a problem when you larger than 5 columns in your results.

In order to make it work for that scenario, only `magiq_include/utils.h` needs to be changed. 
Let's assume you have 6 columns in final results. You need to add following lines to that file:

```
typedef struct Int6 { IndexType x[6]; } Int6;

int compInt6A(const void *elem1, const void *elem2)...;
int compInt6B(const void *elem1, const void *elem2)...;
int compInt6C(const void *elem1, const void *elem2)...;
int compInt6D(const void *elem1, const void *elem2)...;
int compInt6E(const void *elem1, const void *elem2)...;
int compInt6F(const void *elem1, const void *elem2)...;

int (*comp[24])(const void *, const void *);

void initComp() {
    comp[0] = compInt3A;
    comp[1] = compInt3B;
    comp[2] = compInt3C;

    comp[6] = compInt4A;
    comp[7] = compInt4B;
    comp[8] = compInt4C;
    comp[9] = compInt4D;

    comp[12] = compInt5A;
    comp[13] = compInt5B;
    comp[14] = compInt5C;
    comp[15] = compInt5D;
    comp[16] = compInt5E;

    comp[18] = compInt6A;
    comp[19] = compInt6B;
    comp[20] = compInt6C;
    comp[21] = compInt6D;
    comp[22] = compInt6E;
    comp[23] = compInt6F;
}
```

This way is not efficient and scalable, needs to be improved in the future.