import jsPDF from 'jspdf';

export const generateFraudReport = async (prediction, selectedFile) => {
  return new Promise((resolve, reject) => {
    try {
      const pdf = new jsPDF();
      let yPosition = 20;
      
      // Helper function to add image to PDF
      const addImageToPdf = (imageData) => {
        try {
          // Add image on the right side
          pdf.addImage(imageData, 'JPEG', 110, 20, 80, 60);
          pdf.setFontSize(10);
          pdf.setTextColor(80, 80, 80);
          pdf.text('Analyzed Image', 110, 85);
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
            
            // Continue with PDF generation after image is loaded
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
        // Header Section
        pdf.setFillColor(59, 130, 246); // Blue background
        pdf.rect(0, 0, 210, 35, 'F');
        
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(24);
        pdf.text('🛡️ FRAUD DETECTION REPORT', 20, 20);
        
        pdf.setFontSize(12);
        pdf.text('Insurance Claim Analysis', 20, 28);
        
        yPosition = 50;

        // Report Information
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('Report Information', 20, yPosition);
        yPosition += 10;
        
        pdf.setFontSize(10);
        pdf.setTextColor(80, 80, 80);
        pdf.text(`Generated: ${new Date().toLocaleString()}`, 20, yPosition);
        yPosition += 5;
        pdf.text(`File: ${selectedFile.name}`, 20, yPosition);
        yPosition += 5;
        pdf.text(`Size: ${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`, 20, yPosition);
        yPosition += 15;

        // Main Analysis Results
        pdf.setFillColor(248, 250, 252); // Light gray background
        pdf.rect(15, yPosition - 5, 95, 25, 'F');
        
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(16);
        pdf.text('📊 ANALYSIS RESULTS', 20, yPosition + 5);
        yPosition += 15;
        
        // Prediction with colored background
        const predictionColor = prediction.prediction === 'FRAUD' ? [239, 68, 68] : [34, 197, 94];
        pdf.setFillColor(...predictionColor);
        pdf.rect(20, yPosition, 85, 8, 'F');
        
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(14);
        pdf.text(`PREDICTION: ${prediction.prediction}`, 22, yPosition + 6);
        yPosition += 15;
        
        // Confidence and metrics
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(12);
        pdf.text(`Confidence Level: ${prediction.confidence}%`, 20, yPosition);
        yPosition += 7;
        pdf.text(`Risk Category: ${prediction.riskLevel}`, 20, yPosition);
        yPosition += 15;

        // Risk Assessment Section
        pdf.setFillColor(254, 243, 199); // Yellow background
        pdf.rect(15, yPosition - 5, 180, 30, 'F');
        
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('⚠️ RISK ASSESSMENT', 20, yPosition + 5);
        yPosition += 12;
        
        pdf.setFontSize(11);
        pdf.text(`Risk Level: ${prediction.riskLevel}`, 20, yPosition);
        yPosition += 6;
        
        const actionLines = pdf.splitTextToSize(`Recommended Action: ${prediction.recommendedAction}`, 160);
        pdf.text(actionLines, 20, yPosition);
        yPosition += actionLines.length * 5 + 10;

        // Probability Breakdown
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('📈 PROBABILITY BREAKDOWN', 20, yPosition);
        yPosition += 10;
        
        pdf.setFontSize(11);
        pdf.setTextColor(220, 38, 38); // Red for fraud
        pdf.text(`Fraud Probability: ${prediction.fraudProbability}%`, 20, yPosition);
        yPosition += 6;
        
        pdf.setTextColor(34, 197, 94); // Green for non-fraud
        pdf.text(`Non-Fraud Probability: ${prediction.nonFraudProbability}%`, 20, yPosition);
        yPosition += 15;

        // Technical Details
        pdf.setFillColor(241, 245, 249); // Light blue background
        pdf.rect(15, yPosition - 5, 180, 35, 'F');
        
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('🔧 TECHNICAL DETAILS', 20, yPosition + 5);
        yPosition += 12;
        
        pdf.setFontSize(10);
        pdf.setTextColor(80, 80, 80);
        pdf.text(`Processing Time: ${prediction.processingTime || 'N/A'}`, 20, yPosition);
        yPosition += 5;
        pdf.text('Model Architecture: EfficientNet-B1', 20, yPosition);
        yPosition += 5;
        pdf.text('Input Resolution: 224x224 pixels', 20, yPosition);
        yPosition += 5;
        pdf.text('Analysis Date: ' + new Date().toLocaleDateString(), 20, yPosition);
        yPosition += 15;

        // Model Performance Metrics
        pdf.setTextColor(40, 40, 40);
        pdf.setFontSize(14);
        pdf.text('📊 MODEL PERFORMANCE', 20, yPosition);
        yPosition += 10;
        
        pdf.setFontSize(11);
        pdf.setTextColor(80, 80, 80);
        pdf.text('• Model: Fast Precision EfficientNet-B1', 20, yPosition);
        yPosition += 6;
        pdf.text('• Overall Accuracy: 91.4%', 20, yPosition);
        yPosition += 6;
        pdf.text('• Precision: 87.9%', 20, yPosition);
        yPosition += 6;
        pdf.text('• Recall: 86.0%', 20, yPosition);
        yPosition += 6;
        pdf.text('• Training Dataset: 10,000+ insurance claim images', 20, yPosition);
        yPosition += 15;

        // Footer
        pdf.setFillColor(71, 85, 105); // Dark blue
        pdf.rect(0, 277, 210, 20, 'F');
        
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(10);
        pdf.text('FraudGuard AI - Advanced Insurance Fraud Detection System', 20, 285);
        pdf.text('Confidential Report - For Internal Use Only', 20, 292);

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