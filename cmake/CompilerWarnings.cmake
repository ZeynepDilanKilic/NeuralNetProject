# Defines the INTERFACE target neuralnet::warnings. Link it PRIVATELY into every target
# owned by this project; third-party code (e.g. Catch2) is deliberately left untouched.

add_library(neuralnet_warnings INTERFACE)
add_library(neuralnet::warnings ALIAS neuralnet_warnings)

if(MSVC)
    target_compile_options(neuralnet_warnings INTERFACE
        /W4          # baseline
        /permissive- # standards conformance
        /w14242      # conversion, possible loss of data
        /w14254      # operator: conversion from 'type1:field_bits' to 'type2:field_bits'
        /w14263      # member function does not override any base class virtual member function
        /w14265      # class has virtual functions, but destructor is not virtual
        /w14287      # unsigned/negative constant mismatch
        /w14296      # expression is always true/false
        /w14311      # pointer truncation
        /w14545 /w14546 /w14547 /w14549 /w14555 # suspicious expressions / unused values
        /w14619      # pragma warning: there is no warning number 'number'
        /w14640      # thread-unsafe static member initialization
        /w14826      # sign-extended conversion
        /w14905 /w14906 # string literal cast
        /w14928      # illegal copy-initialization
        /utf-8)
    if(NEURALNET_WARNINGS_AS_ERRORS)
        target_compile_options(neuralnet_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(neuralnet_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
        -Wnon-virtual-dtor)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(neuralnet_warnings INTERFACE
            -Wmisleading-indentation
            -Wduplicated-cond
            -Wduplicated-branches
            -Wlogical-op
            -Wuseless-cast)
    endif()
    if(NEURALNET_WARNINGS_AS_ERRORS)
        target_compile_options(neuralnet_warnings INTERFACE -Werror)
    endif()
endif()
