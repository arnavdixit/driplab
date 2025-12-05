import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Fashion AI App',
  description: 'Wardrobe management and outfit recommendation app',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
