# **About the website** 

## 🎯 **Section 1: Biological Inspiration**

- **Visual simulation** of dogs finding food using their sense of smell
- Shows scent gradient visualization (green glow around food)
- Dogs exhibit realistic behaviors: following gradients, zigzagging when lost, cooperation
- Real-time stats: number of dogs searching, average distance to food, dogs that found food

## ⚙️ **Section 2: Algorithm Visualization**

- **Two tabs** for organizing parameters:

  - **Basic Parameters**: Pack configuration, momentum, attraction weights, exploration
  - **Advanced Parameters**: Zigzag behavior, elitist exchange, gradient refinement toggles
  
- **Interactive controls** for all hyperparameters with real-time value displays
- **Visual simulation** showing multiple packs (different colors) converging to target
- **Four action buttons**:

  - Run Optimization
  - Reset
  - Random Hyperparameters (test random configurations)
  - Find Best Hyperparameters (auto-tune)
  
- **Real-time statistics**: iteration count, best fitness, evaluations, convergence rate

## 🏔️ **Section 3: Rosenbrock Function Optimization**

- **3D colorful visualization** of the Rosenbrock function surface
- Color-coded height map (red = high values, blue/green = low values)
- **Interactive 3D view controls**: rotate angle and elevation to see from different perspectives
- Green marker shows global minimum at (1, 1)
- Dogs shown as colored dots on the surface
- Golden ring highlights current best position
- **Statistics tracked**: best position found, function value, distance to optimum, success rate
- Auto-tune button optimized specifically for Rosenbrock function characteristics
