import jsPDF from 'jspdf';

export const generateFraudReport = async (prediction, selectedFile) => {
  return new Promise((resolve, reject) => {
    try {
      const pdf = new jsPDF();
      
      // Helper function to add image to PDF
      const addImageToPdf = (imageData) => {
        try {
          // Add image on the right side with better positioning
          pdf.addImage(imageData, 'JPEG', 130, 50, 60, 45);
          pdf.setFontSize(9);
          pdf.setTextColor(100, 100, 100);
          pdf.text('Claim Image', 130, 100);
        } catch (error) {
          console.warn('Could not add image to PDF:', error);
        }
      };

      // Add image if available
      if (selectedFile && selectedFile.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
          try {
            addImageToPdf(e.target.result);
            generatePdfContent();
          } catch (error) {
            console.warn('Image loading failed, continuing without image');
            generatePdfContent();
          }
        };
        reader.onerror = () => {
          console.warn('Image reading failed, continuing without image');
          generatePdfContent();
        };
        reader.readAsDataURL(selectedFile);
      } else {
        generatePdfContent();
      }

      function generatePdfContent() {
        let yPos = 20;

        // Header Section with better design
        pdf.setFillColor(24, 60, 120); // Professional dark blue
        pdf.rect(0, 0, 210, 40, 'F');
        
        // Title
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(22);
        pdf.text('INSURANCE FRAUD DETECTION REPORT', 20, 20);
        
        // Subtitle
        pdf.setFontSize(11);
        pdf.text('AI-Powered Claim Analysis', 20, 30);
        
        // Report ID and Date
        pdf.setFontSize(9);
        const reportId = `FR-${Date.now().toString().slice(-6)}`;
        pdf.text(`Report ID: ${reportId}`, 140, 20);
        pdf.text(`Generated: ${new Date().toLocaleString()}`, 140, 28);
        
        yPos = 55;

        // Executive Summary Box
        pdf.setFillColor(245, 245, 245);
        pdf.rect(15, yPos, 110, 35, 'F');
        pdf.setDrawColor(200, 200, 200);
        pdf.rect(15, yPos, 110, 35, 'S');
        
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('EXECUTIVE SUMMARY', 20, yPos + 8);
        
        // Main prediction result
        const predictionColor = prediction.prediction === 'FRAUD' ? [220, 38, 38] : [22, 163, 74];
        pdf.setTextColor(...predictionColor);
        pdf.setFontSize(16);
        pdf.text(`RESULT: ${prediction.prediction}`, 20, yPos + 18);
        
        pdf.setTextColor(60, 60, 60);
        pdf.setFontSize(11);
        pdf.text(`Confidence: ${prediction.confidence}%`, 20, yPos + 26);
        pdf.text(`Risk Level: ${prediction.riskLevel}`, 20, yPos + 32);
        
        yPos += 50;

        // Claim Information Section
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('CLAIM INFORMATION', 20, yPos);
        yPos += 8;
        
        pdf.setFontSize(10);
        pdf.setTextColor(80, 80, 80);
        pdf.text(`File Name: ${selectedFile.name}`, 25, yPos);
        yPos += 6;
        pdf.text(`File Size: ${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`, 25, yPos);
        yPos += 6;
        pdf.text(`File Type: ${selectedFile.type}`, 25, yPos);
        yPos += 6;
        pdf.text(`Analysis Date: ${new Date().toLocaleDateString()}`, 25, yPos);
        yPos += 15;

        // Risk Assessment Section
        pdf.setFillColor(255, 245, 235); // Light orange background
        pdf.rect(15, yPos, 180, 25, 'F');
        pdf.setDrawColor(255, 165, 0);
        pdf.rect(15, yPos, 180, 25, 'S');
        
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('RISK ASSESSMENT', 20, yPos + 8);
        
        pdf.setFontSize(11);
        pdf.setTextColor(60, 60, 60);
        pdf.text(`Risk Category: ${prediction.riskLevel}`, 20, yPos + 16);
        
        // Recommended action with proper text wrapping
        const actionText = `Action Required: ${prediction.recommendedAction}`;
        const actionLines = pdf.splitTextToSize(actionText, 160);
        pdf.text(actionLines, 20, yPos + 22);
        
        yPos += 35;

        // Detailed Analysis Section
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('DETAILED ANALYSIS', 20, yPos);
        yPos += 10;
        
        // Probability breakdown with visual bars
        pdf.setFontSize(12);
        pdf.text('Probability Breakdown:', 20, yPos);
        yPos += 8;
        
        // Fraud probability bar
        pdf.setFontSize(10);
        pdf.setTextColor(220, 38, 38);
        pdf.text(`Fraud: ${prediction.fraudProbability}%`, 25, yPos);
        
        // Visual bar for fraud probability
        const fraudWidth = (parseFloat(prediction.fraudProbability) / 100) * 50;
        pdf.setFillColor(220, 38, 38);
        pdf.rect(70, yPos - 3, fraudWidth, 4, 'F');
        pdf.setDrawColor(200, 200, 200);
        pdf.rect(70, yPos - 3, 50, 4, 'S');
        yPos += 8;
        
        // Non-fraud probability bar
        pdf.setTextColor(22, 163, 74);
        pdf.text(`Non-Fraud: ${prediction.nonFraudProbability}%`, 25, yPos);
        
        const nonFraudWidth = (parseFloat(prediction.nonFraudProbability) / 100) * 50;
        pdf.setFillColor(22, 163, 74);
        pdf.rect(70, yPos - 3, nonFraudWidth, 4, 'F');
        pdf.setDrawColor(200, 200, 200);
        pdf.rect(70, yPos - 3, 50, 4, 'S');
        yPos += 15;

        // Technical Specifications
        pdf.setFillColor(240, 248, 255); // Light blue background
        pdf.rect(15, yPos, 180, 30, 'F');
        pdf.setDrawColor(100, 149, 237);
        pdf.rect(15, yPos, 180, 30, 'S');
        
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(12);
        pdf.text('TECHNICAL SPECIFICATIONS', 20, yPos + 8);
        
        pdf.setFontSize(9);
        pdf.setTextColor(80, 80, 80);
        pdf.text(`• Processing Time: ${prediction.processingTime || 'N/A'}`, 25, yPos + 16);
        pdf.text(`• Model Architecture: EfficientNet-B1`, 25, yPos + 21);
        pdf.text(`• Input Resolution: 224x224 pixels`, 25, yPos + 26);
        
        yPos += 40;

        // Model Performance Metrics
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(12);
        pdf.text('MODEL PERFORMANCE METRICS', 20, yPos);
        yPos += 8;
        
        pdf.setFontSize(9);
        pdf.setTextColor(80, 80, 80);
        
        // Create a table-like structure for metrics
        const metrics = [
          ['Model Name:', 'Fast Precision EfficientNet-B1'],
          ['Overall Accuracy:', '91.4%'],
          ['Precision:', '87.9%'],
          ['Recall:', '86.0%'],
          ['F1-Score:', '86.9%'],
          ['Training Dataset:', '10,000+ labeled insurance claim images']
        ];
        
        metrics.forEach((metric, index) => {
          pdf.text(metric[0], 25, yPos + (index * 5));
          pdf.text(metric[1], 80, yPos + (index * 5));
        });
        
        yPos += 35;

        // Disclaimer Section
        pdf.setFillColor(250, 250, 250);
        pdf.rect(15, yPos, 180, 20, 'F');
        pdf.setDrawColor(150, 150, 150);
        pdf.rect(15, yPos, 180, 20, 'S');
        
        pdf.setTextColor(100, 100, 100);
        pdf.setFontSize(8);
        pdf.text('DISCLAIMER: This report is generated by AI and should be used as a supplementary tool.', 20, yPos + 6);
        pdf.text('Final decisions should involve human review and additional verification processes.', 20, yPos + 12);
        pdf.text('For questions or concerns, contact your claims department supervisor.', 20, yPos + 18);

        // Footer
        pdf.setFillColor(24, 60, 120);
        pdf.rect(0, 280, 210, 17, 'F');
        
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(10);
        pdf.text('FraudGuard AI - Advanced Insurance Fraud Detection System', 20, 290);
        pdf.setFontSize(8);
        pdf.text('Confidential and Proprietary - For Internal Use Only', 20, 295);

        // Save the PDF
        const fileName = `fraud-analysis-${selectedFile.name.split('.')[0]}-${Date.now()}.pdf`;
        pdf.save(fileName);
        resolve(fileName);
      }
    } catch (error) {
      reject(error);
    }
  });
};