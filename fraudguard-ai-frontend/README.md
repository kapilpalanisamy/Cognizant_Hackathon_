# FraudGuard AI Frontend

A modern React-based web application for insurance fraud detection using AI.

## Features

- **Modern UI**: Built with React 18 + Vite + Tailwind CSS
- **Component Library**: Shadcn/UI components for professional design
- **File Upload**: Drag & drop image upload with react-dropzone
- **Real-time AI Predictions**: Integration with ML backend API
- **PDF Report Generation**: Automated fraud detection reports with jsPDF
- **Responsive Design**: Mobile-friendly interface
- **Authentication**: Clerk integration (configurable)

## Tech Stack

- **Frontend Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **UI Components**: Shadcn/UI
- **File Upload**: react-dropzone
- **PDF Generation**: jsPDF
- **Authentication**: Clerk
- **HTTP Client**: Fetch API

## Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Set up environment variables**:
   ```bash
   # Copy example env file
   cp .env.example .env.local
   
   # Edit .env.local with your values
   VITE_CLERK_PUBLISHABLE_KEY=your_clerk_key_here
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Build for production**:
   ```bash
   npm run build
   ```

## Environment Variables

- `VITE_CLERK_PUBLISHABLE_KEY`: Clerk authentication key (optional in development)

## API Integration

The frontend communicates with the ML backend API:

- **Development**: `http://localhost:8001`
- **Production**: Configured via Netlify Functions

## Deployment

### Netlify (Recommended)

1. Connect your repository to Netlify
2. Set build command: `npm run build`
3. Set publish directory: `dist`
4. Add environment variables in Netlify dashboard
5. Deploy!

### Alternative Deployment

- **Vercel**: Compatible with Vite React apps
- **Firebase Hosting**: Static hosting option
- **GitHub Pages**: Free static hosting

## Project Structure

```
src/
├── components/          # Reusable UI components
├── pages/              # Page components
├── utils/              # Utility functions
├── api/                # API integration
└── assets/             # Static assets
```

## Key Components

- **fraud-detection.jsx**: Main fraud detection interface
- **pdfGenerator.js**: PDF report generation utility
- **header.jsx**: Navigation and branding
- **ui/**: Shadcn/UI component library

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details
