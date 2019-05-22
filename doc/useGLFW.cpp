//
// Created by hu on 2019/5/22.
//
#include <GLFW/glfw3.h>
int main(void)
{
    GLFWwindow* window;

    /* Initialize the library
     * Before you can use most GLFW functions, the library must be initialized
     * */
    if (!glfwInit())
        return -1;

    /*
     * The window and its OpenGL context are created
     * with a single call to glfwCreateWindow,
     * which returns a handle to the created
     * combined window and context object
     */
    //This creates a 640 by 480 windowed mode window with an OpenGL context.
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /*
     * Before you can use the OpenGL API,
     * you must have a current OpenGL context.
     * The context will remain current
     * until you make another context current
     * or until the window owning the current context is destroyed.
     */
    glfwMakeContextCurrent(window);

    /*
     * When the user attempts to close the window,
     * either by pressing the close widget in the title bar
     * or using a key combination like Alt+F4,
     * this flag is set to 1.
     * Note that the window isn't actually closed,
     * so you are expected to monitor this flag
     * and either destroy the window
     * or give some kind of feedback to the user.
     */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        /*
         * GLFW windows by default use double buffering.
         * That means that each window has two rendering buffers;
         * a front buffer and a back buffer.
         * The front buffer is the one being displayed
         * and the back buffer the one you render to.
         *
         * When the entire frame has been rendered,
         * the buffers need to be swapped with one another,
         * so the back buffer becomes the front buffer and vice versa.
         */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }
    /*
     * When you are done using GLFW,
     * typically just before the application exits,
     * you need to terminate GLFW
     */
    glfwTerminate();
    return 0;
}