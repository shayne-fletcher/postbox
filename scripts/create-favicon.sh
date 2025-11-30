#!/bin/bash
# Create favicon from 📬 emoji
# Requires ImageMagick: brew install imagemagick (Mac) or apt install imagemagick (Linux)

set -e

cd "$(dirname "$0")/.."

echo "Creating favicon from 📬 emoji..."

# Create SVG with the emoji
cat > emoji.svg <<'EOF'
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <text x="50" y="50" font-size="90" text-anchor="middle" dominant-baseline="middle">📬</text>
</svg>
EOF

# Convert SVG to different sizes
convert -background none emoji.svg -resize 32x32 favicon-32x32.png
convert -background none emoji.svg -resize 16x16 favicon-16x16.png

# Combine into .ico file
convert favicon-32x32.png favicon-16x16.png favicon.ico

# Clean up intermediate files
rm emoji.svg favicon-32x32.png favicon-16x16.png

echo "✓ Created favicon.ico from 📬 emoji"
echo "Favicon is ready!"
