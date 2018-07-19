function(resource_wrap_cpp _OUT RESOURCES)
    foreach(RESOURCE ${RESOURCES})
        get_filename_component(FILE ${RESOURCE} NAME_WE)
        get_filename_component(EXT ${RESOURCE} EXT)

        string(REGEX REPLACE "\\." "_" EXT "${EXT}")

        add_custom_command(
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FILE}${EXT}.cpp"
            COMMAND embed_resource ARGS "${FILE}${EXT}" "${RESOURCE}"
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${RESOURCE}
        )

        list(APPEND SOURCE "${CMAKE_CURRENT_BINARY_DIR}/${FILE}${EXT}.cpp")
    endforeach(RESOURCE)

    set(${_OUT} ${SOURCE} PARENT_SCOPE)
endfunction()
