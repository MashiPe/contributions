# Quick Overview on Volume Rendering   
Volumetric rendering is fundamental to NeRFs, as it is the primary technique for rendering 3D scenes by simulating how light interacts with a volume. While it's a broad topic with many applications and techniques, we have a detailed series on volumetric rendering that we highly recommend checking out [here](https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/intro-volume-rendering.html). For this lesson, we’ll provide a quick overview to help you grasp the essentials before diving into the specifics of NeRFs.   
   
Volumetric rendering is a technique in computer graphics used to represent 3D objects and scenes by simulating how light interacts with volumes in space. In this context, a ***volume*** refers to a three-dimensional region that represents an entity, often characterized by properties such as density, opacity, or color. Common examples include clouds, smoke, and fluids. Volumetric rendering often involves accounting for phenomena like transmittance, absorption, particle density, and scattering to simulate the way light interacts with these volumes accurately. To achieve this, we rely on the Beer-Lambert law and the ray-marching algorithm, which help integrate all these interactions into a single visual result.   
   
![image.png](../../images/NeRF/image.png)    
   
In simple terms, volumetric rendering works by sending a beam of light through space and taking samples at regular intervals along its path. At each step, we calculate how the light interacts with the volume at that point. Once enough samples are collected, we combine the results to generate the final pixel.   
   
Now that we have a basic idea on how volumetric rendering works (at least that is what we hope), we can move on to the next lesson where we will explain in detail how we can combine neural networks and volumetric rendering to synthesize novel views of our 3D scene from RGB photos.    
