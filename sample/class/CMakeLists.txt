set(openmpngine_src
    main.cu
    point.cu)

set(openmpngine_header point.hpp)

add_sample(SOURCES ${openmpngine_src} HEADERS ${openmpngine_header})
