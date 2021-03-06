﻿using System;
using System.Diagnostics;
using System.IO;
using OpenTK.Graphics.OpenGL;

namespace Template {

    class Game {
        public static int screenWidth, screenHeight;
        // when GLInterop is set to true, the fractal is rendered directly to an OpenGL texture
        bool GLInterop = true;
        // load the OpenCL program; this creates the OpenCL context
        static OpenCLProgram ocl = new OpenCLProgram("../../program.cl");
        // find the two kernels in the program
        OpenCLKernel kernel = new OpenCLKernel(ocl, "device_function");
        OpenCLKernel swapper = new OpenCLKernel(ocl, "ruiltransactie");
        // initialize two uint arrays
        uint[] pattern, second;
        // initialize the two buffers for simple existance of bits
        OpenCLBuffer<uint> patternB;
        OpenCLBuffer<uint> secondB;
        // create the buffer used for drawing
        OpenCLBuffer<uint> teken = new OpenCLBuffer<uint>(ocl, screenWidth * screenHeight);
        // create an OpenGL texture to which OpenCL can send data
        OpenCLImage<int> image = new OpenCLImage<int>(ocl, screenWidth, screenHeight);
        public Surface screen;
        Stopwatch timer = new Stopwatch();
        uint pw, ph;
        int generation = 0;

        // helper function for setting one bit in the pattern buffer
        void BitSet(uint x, uint y) { second[(int)(y * pw + (x >> 5))] |= 1U << (int)(x & 31); }

        // mouse handling: dragging functionality (copied from Game of Life C# code)
        uint xoffset = 0, yoffset = 0;
        bool lastLButtonState = false;
        int dragXStart, dragYStart, offsetXStart, offsetYStart;
        public void SetMouseState(int x, int y, bool pressed) {
            if (pressed) {
                if (lastLButtonState) {
                    int deltax = x - dragXStart, deltay = y - dragYStart;
                    xoffset = (uint)Math.Min(pw * 32 - screen.width, Math.Max(0, offsetXStart - deltax));
                    yoffset = (uint)Math.Min(ph - screen.height, Math.Max(0, offsetYStart - deltay));
                }
                else {
                    dragXStart = x;
                    dragYStart = y;
                    offsetXStart = (int)xoffset;
                    offsetYStart = (int)yoffset;
                    lastLButtonState = true;
                }
            }
            else lastLButtonState = false;
        }
        
        public void Init() {
            StreamReader sr = new StreamReader("../../data/turing_js_r.rle");
            uint state = 0, n = 0, x = 0, y = 0;
            while (true) {
                String line = sr.ReadLine();
                if (line == null) break; // end of file
                int pos = 0;
                if (line[pos] == '#') continue; // comment line
                else if (line[pos] == 'x') { // header
                    String[] sub = line.Split(new char[] { '=', ',' }, StringSplitOptions.RemoveEmptyEntries);
                    pw = (UInt32.Parse(sub[1]) + 31) / 32;
                    ph = UInt32.Parse(sub[3]);
                    pattern = new uint[pw * ph];
                    second = new uint[pw * ph];
                }
                else {
                    while (pos < line.Length) {
                        Char c = line[pos++];
                        if (state == 0) if (c < '0' || c > '9') { state = 1; n = Math.Max(n, 1); } else n = (uint)(n * 10 + (c - '0'));
                        if (state == 1) // expect other character
                        {
                            if (c == '$') { y += n; x = 0; } // newline
                            else if (c == 'o') for (int i = 0; i < n; i++) BitSet(x++, y); else if (c == 'b') x += n;
                            state = n = 0;
                        }
                    }
                }
            }
            // assign values to the initialized buffers
            patternB = new OpenCLBuffer<uint>(ocl, pattern);
            secondB = new OpenCLBuffer<uint>(ocl, second);
            if (GLInterop) {
                // pass on the arguments only needed in Init() because of GLInterop
                kernel.SetArgument(0, image);
                kernel.SetArgument(1, patternB);
                kernel.SetArgument(2, secondB);
                swapper.SetArgument(0, patternB);
                swapper.SetArgument(1, secondB);
            }
            // pass on the arguments that are only needed once
            kernel.SetArgument(3, pw);
            kernel.SetArgument(4, ph);
            kernel.SetArgument(5, screenWidth);
            kernel.SetArgument(6, screenHeight);
            swapper.SetArgument(2, pw);
            swapper.SetArgument(3, ph);
        }

        public void Tick() {
            // start timer
            timer.Restart();
            GL.Finish();
            // clear the screen
            screen.Clear(0);
            // do opencl stuff
            if (!GLInterop) {
                // pass on the buffers
                kernel.SetArgument(0, patternB);
                kernel.SetArgument(1, secondB);
                kernel.SetArgument(2, teken);
                swapper.SetArgument(0, patternB);
                swapper.SetArgument(1, secondB);
            }
            // pass on the offsets that constantly change
            kernel.SetArgument(7, xoffset);
            kernel.SetArgument(8, yoffset);
            // execute kernel
            long[] workSize = { pw * 32, ph };
            long[] localSize = { 32, 4 };
            if (GLInterop) {
                // INTEROP PATH:
                // Use OpenCL to fill an OpenGL texture; this will be used in the
                // Render method to draw a screen filling quad. This is the fastest
                // option, but interop may not be available on older systems.
                // lock the OpenGL texture for use by OpenCL
                kernel.LockOpenGLObject(image.texBuffer);
                // execute the kernel
                kernel.Execute(workSize, null);
                kernel.StopMaar();
                // execute the kernel which swaps the buffers
                swapper.Execute(workSize, null);
                swapper.StopMaar();
                // unlock the OpenGL texture so it can be used for drawing a quad
                kernel.UnlockOpenGLObject(image.texBuffer);
            }
            else {
                // NO INTEROP PATH:
                // Use OpenCL to fill a C# pixel array, encapsulated in an
                // OpenCLBuffer<int> object (buffer). After filling the buffer, it
                // is copied to the screen surface, so the template code can show
                // it in the window.
                // execute the kernel
                kernel.Execute(workSize, null );
                kernel.StopMaar();
                // execute the kernel which swaps the buffers
                swapper.Execute(workSize, null );
                swapper.StopMaar();
                //swapper.StopMaar();
                // get the data from the device to the host
                teken.CopyFromDevice();
                // plot pixels using the data on the host
                for (int y = 0; y < screenHeight; y++) {
                    for (int x = 0; x < screenWidth; x++) {
                        screen.pixels[x + y * screenWidth] = (int)teken[(x + y * screenWidth)];
                    }
                }
            }
            // report performance
            Console.WriteLine("generation " + generation++ + ": " + timer.ElapsedMilliseconds + "ms");
        }
        public void Render() {
            // use OpenGL to draw a quad using the texture that was filled by OpenCL
            if (GLInterop) {
                GL.LoadIdentity();
                GL.BindTexture(TextureTarget.Texture2D, image.OpenGLTextureID);
                GL.Begin(PrimitiveType.Quads);
                GL.TexCoord2(0.0f, 1.0f); GL.Vertex2(-1.0f, -1.0f);
                GL.TexCoord2(1.0f, 1.0f); GL.Vertex2(1.0f, -1.0f);
                GL.TexCoord2(1.0f, 0.0f); GL.Vertex2(1.0f, 1.0f);
                GL.TexCoord2(0.0f, 0.0f); GL.Vertex2(-1.0f, 1.0f);
                GL.End();
            }
        }
    }

} // namespace Template