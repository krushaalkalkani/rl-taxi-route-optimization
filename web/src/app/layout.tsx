import type { Metadata } from "next";
import { Barlow, Barlow_Condensed } from "next/font/google";
import "./globals.css";

const barlow = Barlow({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-barlow",
});
const barlowCondensed = Barlow_Condensed({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-barlow-condensed",
});

export const metadata: Metadata = {
  title: "Midtown Cab RL: a taxi that learned Manhattan",
  description:
    "A reinforcement-learning taxi dispatcher trained on Midtown Manhattan's real street network, with traffic calibrated against Google Maps. Built with Gymnasium and Stable-Baselines3.",
  openGraph: {
    title: "Midtown Cab RL",
    description: "A reinforcement-learning taxi that learned to work a shift in Midtown Manhattan.",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${barlow.variable} ${barlowCondensed.variable}`}>{children}</body>
    </html>
  );
}
