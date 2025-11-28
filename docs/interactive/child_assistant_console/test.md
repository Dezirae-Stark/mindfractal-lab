# Test Avatar Display

Testing if WebP images load correctly:

## Direct Image Tags

### Neutral
<img src="./graphics/realistic/cytherea_neutral.webp?v=1.0" width="300" alt="Neutral">

### Focused  
<img src="./graphics/realistic/cytherea_focused.webp?v=1.0" width="300" alt="Focused">

### Dream
<img src="./graphics/realistic/cytherea_dream.webp?v=1.0" width="300" alt="Dream">

### Overload
<img src="./graphics/realistic/cytherea_overload.webp?v=1.0" width="300" alt="Overload">

### Celebrate
<img src="./graphics/realistic/cytherea_celebrate.webp?v=1.0" width="300" alt="Celebrate">

## JavaScript Test
<div id="test-container">
    <img id="test-img" src="./graphics/realistic/cytherea_neutral.webp?v=1.0" width="300">
    <br>
    <button onclick="document.getElementById('test-img').src='./graphics/realistic/cytherea_focused.webp?v=1.0'">Focused</button>
    <button onclick="document.getElementById('test-img').src='./graphics/realistic/cytherea_dream.webp?v=1.0'">Dream</button>
</div>