import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import Autoplay from "embla-carousel-autoplay";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Link } from "react-router-dom";
import { Shield, Upload, BarChart3, Download, Brain, Zap } from "lucide-react";

const LandingPage = () => {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Detection",
      description: "Advanced EfficientNet model with 87.9% precision and 91.4% accuracy"
    },
    {
      icon: Zap,
      title: "Fast Processing",
      description: "Get instant fraud predictions in under 3 seconds"
    },
    {
      icon: Download,
      title: "Detailed Reports",
      description: "Generate comprehensive PDF reports with confidence scores"
    }
  ];

  const faqs = [
    {
      question: "How accurate is the fraud detection?",
      answer: "Our AI model achieves 87.9% precision and 91.4% accuracy, making it enterprise-grade reliable for insurance claim assessment."
    },
    {
      question: "What file formats are supported?",
      answer: "We support common image formats including JPG, JPEG, PNG, and WEBP files up to 10MB in size."
    },
    {
      question: "Is my data secure?",
      answer: "Yes, all images are encrypted and processed securely. We do not store images permanently and follow industry-standard security practices."
    },
    {
      question: "Can I download the analysis results?",
      answer: "Absolutely! You can download detailed PDF reports containing the analysis results, confidence scores, and recommendations."
    }
  ];

  return (
    <main className="flex flex-col gap-10 sm:gap-20 py-10 sm:py-20">
      <section className="text-center">
        <h1 className="flex flex-col items-center justify-center gradient-title font-extrabold text-4xl sm:text-6xl lg:text-8xl tracking-tighter py-4">
          AI-Powered Fraud Detection
          <span className="flex items-center gap-2 sm:gap-6">
            for Insurance Claims
            <Shield className="h-14 sm:h-24 lg:h-32 text-blue-600" />
          </span>
        </h1>
        <p className="text-gray-300 sm:mt-4 text-xs sm:text-xl">
          Upload vehicle damage images and get instant AI-powered fraud analysis with detailed reports
        </p>
      </section>
      
      <div className="flex gap-6 justify-center">
        <Link to="/fraud-detection">
          <Button variant="default" size="xl" className="bg-blue-600 hover:bg-blue-700">
            <Upload className="mr-2 h-5 w-5" />
            Start Analysis
          </Button>
        </Link>
        <Link to="/dashboard">
          <Button variant="outline" size="xl">
            <BarChart3 className="mr-2 h-5 w-5" />
            View Dashboard
          </Button>
        </Link>
      </div>

      {/* Features Section */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {features.map((feature, index) => (
          <Card key={index} className="text-center">
            <CardHeader>
              <feature.icon className="h-12 w-12 mx-auto text-blue-600 mb-4" />
              <CardTitle>{feature.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-gray-600">{feature.description}</p>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* Statistics Section */}
      <section className="text-center bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-8">
        <h2 className="text-3xl font-bold mb-8">Model Performance</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <div className="text-4xl font-bold text-blue-600">87.9%</div>
            <div className="text-gray-600">Precision</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-green-600">86.0%</div>
            <div className="text-gray-600">Recall</div>
          </div>
          <div>
            <div className="text-4xl font-bold text-purple-600">91.4%</div>
            <div className="text-gray-600">Accuracy</div>
          </div>
        </div>
      </section>

      {/* How it Works Section */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="font-bold">For Insurance Adjusters</CardTitle>
          </CardHeader>
          <CardContent>
            Upload claim images and get instant AI analysis with confidence scores and risk assessments.
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="font-bold">For Insurance Companies</CardTitle>
          </CardHeader>
          <CardContent>
            Streamline claim processing, reduce manual review time, and minimize fraudulent payouts.
          </CardContent>
        </Card>
      </section>

      {/* FAQ Section */}
      <section>
        <h2 className="text-3xl font-bold text-center mb-8">Frequently Asked Questions</h2>
        <Accordion type="multiple" className="w-full">
          {faqs.map((faq, index) => (
            <AccordionItem key={index} value={`item-${index + 1}`}>
              <AccordionTrigger>{faq.question}</AccordionTrigger>
              <AccordionContent>{faq.answer}</AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </section>
    </main>
  );
};

export default LandingPage;
