## Queries

This document will show how the queries should look like.

A very simple domain specific language for matrix algebra operations in this scenario is implemented.
However, the functionalities are limited because it's only a simple string parser.

There are three lines just for notation,

- Line 1

    Line 1 can be name of this query, you can use any name

- query_execution

    This is the second line, to denote following lines are for query execution

    `G` : sparse matrix representation of original rdf graph data 
    
    `m_x_x` : the name of intermediate matrices

    `m_x_x.D` : row reduction of matrix

    `m_x_x.T` : transpose of matrix

    `I^xxxx(pos)*xxxx(val)` : sparse matrix only has value(`val`) in position(`(pos, pos)`), if `val` is not given, it's `1` 

    `⊗` : sermiring multiplication (sermiring : `LOS.ISEQ`)

    `×` : normal multiplication (sermiring : `TIMES.PLUS`)

- result_generation

    This is after query commands, to denote following lines are for result generation

    `join` : perform join operation, like table inner join, for example, join `m_0_4` with `m_3_4` will generate `m_0_3_4`, the second table should be 2 columns at all time, and the joint column should always be the second column of the second table 

    `filter` : perform filter operation, remove some entries from the first table from second table, two tables should have one common column and one renaing rule, for example, filter `m_2_3_4_5` with `m_6_4(6=5)` will keep all entries that `m_2_3_4_5.col4 == m_6_4.col4 && m_2_3_4_5.col5 == m_6_4.col6`

### Examples of LUBM benchmark queries (there queries have hard-coded constants, they are for special generated LUBM dataset)

1. L1

    ```
    LUBM_Q1
    query_execution
    m_4_0 = G ⊗ I^103594630*17
    m_3_4 = G ⊗ m_4_0.D*12
    m_2_3 = G.T ⊗ m_3_4.D*17
    m_2_3 = I^139306106 × m_2_3
    m_5_3 = G.T ⊗ m_2_3.T.D*14
    m_1_5 = G.T ⊗ m_5_3.D*17
    m_1_5 = I^130768016 × m_1_5
    m_6_5 = G.T ⊗ m_1_5.T.D*3
    m_6_5 = m_3_4.T.D × m_6_5
    m_3_4 = m_3_4 × m_6_5.D
    m_5_3 = m_6_5.T.D × m_5_3
    m_3_4 = m_5_3.T.D × m_3_4
    m_4_0 = m_3_4.T.D × m_4_0
    result_generation
    join%m_4_0
    join%m_3_4
    join%m_5_3
    filter%6=4%m_6_5
    join%m_1_5
    join%m_2_3
    ```

2. L2

    ```
    LUBM_Q2
    query_execution
    m_1_0 = G ⊗ I^235928023*17
    m_2_1 = G.T ⊗ m_1_0.D*9
    m_1_0 = m_2_1.T.D × m_1_0
    result_generation
    join%m_1_0
    join%m_2_1
    ```

3. L3

    L3 will return empty result, so you don't need to give it full commands, actually the program will stop after 4 line executions

    ```
    LUBM_Q3
    query_execution
    m_4_0 = G ⊗ I^103594630*17
    m_3_4 = G ⊗ m_4_0.D*12
    m_2_3 = G.T ⊗ m_3_4.D*17
    m_2_3 = I^223452631 × m_2_3

    m_5_3 = G ⊗ m_2_3.T.D*14
    m_1_5 = G.T ⊗ m_5_3.D*17
    m_1_5 = I^130768016 × m_1_5
    m_6_5 = G.T ⊗ m_1_5.T.D*3
    m_6_5 = m_3_4.T.D × m_6_5
    m_3_4 = m_3_4 × m_6_5.D
    m_5_3 = m_6_5.T.D × m_5_3
    m_3_4 = m_5_3.T.D × m_3_4
    m_4_0 = m_3_4.T.D × m_4_0
    ```

4. L4

    ```
    LUBM_Q4
    query_execution
    m_2_0 = G ⊗ I^2808777*7
    m_1_2 = G.T ⊗ m_2_0.D*17
    m_1_2 = I^291959481 × m_1_2
    m_3_2 = G.T ⊗ m_1_2.T.D*9
    m_4_2 = G.T ⊗ m_3_2.T.D*8
    m_5_2 = G.T ⊗ m_4_2.T.D*2
    m_2_0 = m_5_2.T.D ⊗ m_2_0
    result_generation
    join%m_2_0
    join%m_5_2
    join%m_4_2
    join%m_3_2
    join%m_1_2
    ```

5. L5

    ```
    LUBM_Q1
    query_execution
    m_2_0 = G ⊗ I^191176245*17
    m_1_2 = G.T ⊗ m_2_0.D*3
    m_1_2 = I^2808777 × m_1_2
    m_2_0 = m_1_2.T.D ⊗ m_2_0
    result_generation
    join%m_2_0
    join%m_1_2
    ```

6. L6

    ```
    LUBM_Q1
    query_execution
    m_4_0 = G ⊗ I^130768016*17
    m_1_4 = G.T ⊗ m_4_0.D*3
    m_1_4 = I^267261320 × m_1_4
    m_3_4 = G ⊗ m_1_4.T.D*7
    m_2_3 = G.T ⊗ m_3_4.D*17
    m_2_3 = I^291959481 × m_2_3
    m_3_4 = m_2_3.T.D ⊗ m_3_4
    m_4_0 = m_3_4.T.D ⊗ m_4_0
    result_generation
    join%m_4_0
    join%m_3_4
    join%m_2_3
    join%m_1_4
    ```

7. L7

    ```
    LUBM_Q7
    query_execution
    m_5_2 = G ⊗ I^291959481*17
    m_3_5 = G ⊗ m_5_2.D*18
    m_0_3 = G.T ⊗ m_3_5.D*17
    m_0_3 = I^223452631 × m_0_3
    m_4_3 = G.T ⊗ m_0_3.T.D*4
    m_1_4 = G.T ⊗ m_4_3.D*17
    m_1_4 = I^235928023 × m_1_4
    m_6_4 = G ⊗ m_1_4.T.D*5
    m_6_4 = m_3_5.T.D × m_6_4
    m_3_5 = m_3_5 × m_6_4.D
    m_4_3 = m_6_4.T.D × m_4_3
    m_3_5 = m_4_3.T.D × m_3_5
    m_5_2 = m_3_5.T.D × m_5_2
    result_generation
    join%m_5_2
    join%m_3_5
    join%m_4_3
    filter%6=5%m_6_4
    join%m_1_4
    join%m_0_3
    ```