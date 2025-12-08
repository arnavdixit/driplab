# Fashion App Frontend

Next.js frontend for the Fashion AI wardrobe management app.

## Prerequisites

- Node.js 20+ (use `.nvmrc` to ensure correct version)
- npm or yarn

## Setup

1. **Switch to the correct Node.js version:**
   ```bash
   nvm use
   # or
   nvm use 20
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Open [http://localhost:3000](http://localhost:3000)** in your browser.

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Environment Variables

Create a `.env.local` file in the frontend directory:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Manual test checklist (FE-002 upload)

- Drag a JPEG/PNG/WebP into the drop zone and see it listed.
- Click the drop zone to browse and select multiple images.
- Invalid type or >10MB file shows an error notice.
- Preview grid shows thumbnails and allows removal before upload.
- Start upload shows progress and ends with success or error per file.
