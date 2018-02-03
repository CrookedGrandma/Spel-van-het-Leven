using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using System.IO;
using Cloo;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;

namespace Template {

    class Game {
        // when GLInterop is set to true, the fractal is rendered directly to an OpenGL texture
        bool GLInterop = false;
        // load the OpenCL program; this creates the OpenCL context
        static OpenCLProgram ocl = new OpenCLProgram("../../program.cl");
        // find the kernel named 'device_function' in the program
        OpenCLKernel kernel = new OpenCLKernel(ocl, "device_function");
        // create a regular buffer; by default this resides on both the host and the device
        OpenCLBuffer<uint> pattern = new OpenCLBuffer<uint>(ocl, 512 * 512);
        OpenCLBuffer<uint> second = new OpenCLBuffer<uint>(ocl, 512 * 512);
        // create an OpenGL texture to which OpenCL can send data
        OpenCLImage<int> image = new OpenCLImage<int>(ocl, 512, 512);
        public Surface screen;
        Stopwatch timer = new Stopwatch();
        uint pw, ph;

        // helper function for setting one bit in the pattern buffer
        void BitSet(uint x, uint y) { pattern[(int)(y * pw + (x >> 5))] |= 1U << (int)(x & 31); }
        // helper function for getting one bit from the secondary pattern buffer
        uint GetBit(uint x, uint y) { return (second[(int)(y * pw + (x >> 5))] >> (int)(x & 31)) & 1U; }

        // minimalistic .rle file reader for Golly files (see http://golly.sourceforge.net)
        public void Init() {
            StreamReader sr = new StreamReader("../../data/turing_js_r.rle");
            uint state = 0, n = 0, x = 0, y = 0;
            while (true) {
                String line = sr.ReadLine();
                if (line == null) break; // end of file
                int pos = 0;
                if (line[pos] == '#') continue; /* comment line */
                else if (line[pos] == 'x') { // header
                    String[] sub = line.Split(new char[] { '=', ',' }, StringSplitOptions.RemoveEmptyEntries);
                    pw = (UInt32.Parse(sub[1]) + 31) / 32;
                    ph = UInt32.Parse(sub[3]);
                }
                else while (pos < line.Length) {
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
            // swap buffers
            for (int i = 0; i < pw * ph; i++) second[i] = pattern[i];
        }
        public void Tick() {
            GL.Finish();
            // clear the screen
            screen.Clear(0);
            // do opencl stuff
            if (GLInterop) kernel.SetArgument(0, image);
            else {
                kernel.SetArgument(0, pattern);
                kernel.SetArgument(1, second);
                kernel.SetArgument(2, pw);
                kernel.SetArgument(3, ph);
            }
            // execute kernel
            long[] workSize = { 512, 512 };
            long[] localSize = { 32, 4 };
            if (GLInterop) {
                // INTEROP PATH:
                // Use OpenCL to fill an OpenGL texture; this will be used in the
                // Render method to draw a screen filling quad. This is the fastest
                // option, but interop may not be available on older systems.
                // lock the OpenGL texture for use by OpenCL
                kernel.LockOpenGLObject(image.texBuffer);
                // execute the kernel
                kernel.Execute(workSize, localSize);
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
                kernel.Execute(workSize, localSize);
                // get the data from the device to the host
                pattern.CopyFromDevice();
                // plot pixels using the data on the host
                for (int y = 0; y < 512; y++) for (int x = 0; x < 512; x++) {
                        screen.pixels[x + y * screen.width] = (int)pattern[(x + y * 512)];
                    }

                // swap buffers
                for (int i = 0; i < pw * ph; i++) second[i] = pattern[i];
            }
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