import jsPDF from 'jspdf';

export const generateFraudReport = async (prediction, selectedFile) => {
  return new Promise((resolve, reject) => {
    try {
      const pdf = new jsPDF();
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 20;
      const contentWidth = pageWidth - (margin * 2);
      
      // Helper function to add image to PDF with proper alignment
      const addImageToPdf = (imageData) => {
        try {
          // Calculate image position (right side, properly aligned)
          const imageWidth = 55;
          const imageHeight = 40;
          const imageX = pageWidth - margin - imageWidth;
          const imageY = 60; // Adjusted to avoid header overlap
          
          pdf.addImage(imageData, 'JPEG', imageX, imageY, imageWidth, imageHeight);
          pdf.setFontSize(8);
          pdf.setTextColor(100, 100, 100);
          // Center the label under the image
          const labelWidth = pdf.getTextWidth('Uploaded Claim Image');
          pdf.text('Uploaded Claim Image', imageX + (imageWidth - labelWidth) / 2, imageY + imageHeight + 8);
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
        let yPos = 25;

        // Header Section with perfect alignment
        pdf.setFillColor(30, 58, 138); // Professional navy blue
        pdf.rect(0, 0, pageWidth, 50, 'F');
        
        
        
        // Report metadata - top right aligned (positioned first to avoid overlap)
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(8);
        const reportId = `FR-${Date.now().toString().slice(-6)}`;
        const reportIdText = `Report ID: ${reportId}`;
        const dateText = `Generated: ${new Date().toLocaleString()}`;
        
        pdf.text(reportIdText, pageWidth - margin - pdf.getTextWidth(reportIdText), 15);
        pdf.text(dateText, pageWidth - margin - pdf.getTextWidth(dateText), 22);
        
        // Title - perfectly centered
        pdf.setFontSize(18);
        const titleText = 'INSURANCE FRAUD DETECTION REPORT';
        const titleWidth = pdf.getTextWidth(titleText);
        pdf.text(titleText, (pageWidth - titleWidth) / 2, 25);
        
        // Subtitle - centered
        pdf.setFontSize(10);
        const subtitleText = 'AI-Powered Claim Analysis & Risk Assessment';
        const subtitleWidth = pdf.getTextWidth(subtitleText);
        pdf.text(subtitleText, (pageWidth - subtitleWidth) / 2, 35);
        
        yPos = 65;

        // Executive Summary Box - properly aligned
        const summaryBoxHeight = 40;
        const summaryBoxWidth = contentWidth * 0.6; // 60% of content width
        
        pdf.setFillColor(248, 250, 252);
        pdf.setDrawColor(203, 213, 225);
        pdf.setLineWidth(1);
        pdf.rect(margin, yPos, summaryBoxWidth, summaryBoxHeight, 'FD');
        
        pdf.setTextColor(30, 41, 59);
        pdf.setFontSize(14);
        pdf.text('EXECUTIVE SUMMARY', margin + 10, yPos + 12);
        
        // Main prediction result with proper spacing
        const predictionColor = prediction.prediction === 'FRAUD' ? [220, 38, 38] : [22, 163, 74];
        pdf.setTextColor(...predictionColor);
        pdf.setFontSize(18);
        pdf.text(`RESULT: ${prediction.prediction}`, margin + 10, yPos + 24);
        
        // Confidence and risk level - aligned
        pdf.setTextColor(71, 85, 105);
        pdf.setFontSize(11);
        pdf.text(`Confidence: ${prediction.confidence}%`, margin + 10, yPos + 32);
        pdf.text(`Risk Level: ${prediction.riskLevel}`, margin + 10, yPos + 38);
        
        yPos += summaryBoxHeight + 20;

        // Two-column layout for better organization
        const leftColumnWidth = contentWidth * 0.48;
        const rightColumnWidth = contentWidth * 0.48;
        const columnGap = contentWidth * 0.04;
        const leftColumnX = margin;
        const rightColumnX = margin + leftColumnWidth + columnGap;

        // Left Column: Claim Information
        pdf.setFillColor(239, 246, 255);
        pdf.setDrawColor(147, 197, 253);
        pdf.rect(leftColumnX, yPos, leftColumnWidth, 60, 'FD');
        
        pdf.setTextColor(30, 58, 138);
        pdf.setFontSize(13);
        pdf.text('CLAIM INFORMATION', leftColumnX + 8, yPos + 12);
        
        pdf.setFontSize(10);
        pdf.setTextColor(71, 85, 105);
        
        const claimInfo = [
          ['File Name:', selectedFile?.name || 'N/A'],
          ['File Size:', selectedFile ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB` : 'N/A'],
          ['File Type:', selectedFile?.type || 'N/A'],
          ['Analysis Date:', new Date().toLocaleDateString()],
          ['Processing Time:', prediction.processingTime || 'N/A']
        ];
        
        claimInfo.forEach((info, index) => {
          const yOffset = yPos + 22 + (index * 7);
          pdf.text(info[0], leftColumnX + 8, yOffset);
          pdf.text(info[1], leftColumnX + 60, yOffset);
        });

        // Right Column: Risk Assessment
        pdf.setFillColor(254, 242, 242);
        pdf.setDrawColor(252, 165, 165);
        pdf.rect(rightColumnX, yPos, rightColumnWidth, 60, 'FD');
        
        pdf.setTextColor(185, 28, 28);
        pdf.setFontSize(13);
        pdf.text('RISK ASSESSMENT', rightColumnX + 8, yPos + 12);
        
        pdf.setFontSize(10);
        pdf.setTextColor(71, 85, 105);
        
        const riskInfo = [
          ['Category:', prediction.riskLevel],
          ['Fraud Probability:', `${prediction.fraudProbability}%`],
          ['Non-Fraud Probability:', `${prediction.nonFraudProbability}%`],
          ['Confidence Score:', `${prediction.confidence}%`]
        ];
        
        riskInfo.forEach((info, index) => {
          const yOffset = yPos + 22 + (index * 7);
          pdf.text(info[0], rightColumnX + 8, yOffset);
          pdf.text(info[1], rightColumnX + 60, yOffset);
        });
        
        yPos += 80;

        // Recommended Actions Section - full width
        pdf.setFillColor(255, 251, 235);
        pdf.setDrawColor(245, 158, 11);
        pdf.rect(margin, yPos, contentWidth, 25, 'FD');
        
        pdf.setTextColor(180, 83, 9);
        pdf.setFontSize(13);
        pdf.text('RECOMMENDED ACTIONS', margin + 8, yPos + 12);
        
        pdf.setFontSize(10);
        pdf.setTextColor(71, 85, 105);
        const actionText = prediction.recommendedAction || 'Review claim through standard verification process.';
        const actionLines = pdf.splitTextToSize(actionText, contentWidth - 20);
        pdf.text(actionLines, margin + 8, yPos + 20);
        
        yPos += 35;

        // Detailed Analysis Section with visual bars
        pdf.setTextColor(30, 41, 59);
        pdf.setFontSize(14);
        pdf.text('DETAILED PROBABILITY ANALYSIS', margin, yPos);
        yPos += 15;
        
        // Create professional probability bars
        const barWidth = contentWidth * 0.7;
        const barHeight = 12;
        
        // Fraud probability bar
        pdf.setFontSize(11);
        pdf.setTextColor(220, 38, 38);
        pdf.text(`Fraud Detection: ${prediction.fraudProbability}%`, margin, yPos);
        
        // Background bar
        pdf.setFillColor(243, 244, 246);
        pdf.rect(margin, yPos + 3, barWidth, barHeight, 'F');
        
        // Filled bar
        const fraudPercentage = parseFloat(prediction.fraudProbability) / 100;
        const fraudBarWidth = barWidth * fraudPercentage;
        pdf.setFillColor(220, 38, 38);
        pdf.rect(margin, yPos + 3, fraudBarWidth, barHeight, 'F');
        
        // Border
        pdf.setDrawColor(209, 213, 219);
        pdf.rect(margin, yPos + 3, barWidth, barHeight, 'S');
        
        yPos += 25;
        
        // Non-fraud probability bar
        pdf.setTextColor(22, 163, 74);
        pdf.text(`Non-Fraud Detection: ${prediction.nonFraudProbability}%`, margin, yPos);
        
        // Background bar
        pdf.setFillColor(243, 244, 246);
        pdf.rect(margin, yPos + 3, barWidth, barHeight, 'F');
        
        // Filled bar
        const nonFraudPercentage = parseFloat(prediction.nonFraudProbability) / 100;
        const nonFraudBarWidth = barWidth * nonFraudPercentage;
        pdf.setFillColor(22, 163, 74);
        pdf.rect(margin, yPos + 3, nonFraudBarWidth, barHeight, 'F');
        
        // Border
        pdf.setDrawColor(209, 213, 219);
        pdf.rect(margin, yPos + 3, barWidth, barHeight, 'S');
        
        yPos += 30;

        // Technical Specifications in a clean table format
        pdf.setFillColor(248, 250, 252);
        pdf.setDrawColor(203, 213, 225);
        pdf.rect(margin, yPos, contentWidth, 45, 'FD');
        
        pdf.setTextColor(30, 58, 138);
        pdf.setFontSize(13);
        pdf.text('TECHNICAL SPECIFICATIONS', margin + 8, yPos + 12);
        
        pdf.setFontSize(9);
        pdf.setTextColor(71, 85, 105);
        
        const techSpecs = [
          ['Model Architecture:', 'EfficientNet-B1 Deep Learning Model'],
          ['Input Resolution:', '224 × 224 pixels (RGB)'],
          ['Processing Framework:', 'PyTorch with CUDA acceleration'],
          ['Analysis Method:', 'Computer Vision + Pattern Recognition']
        ];
        
        techSpecs.forEach((spec, index) => {
          const yOffset = yPos + 20 + (index * 6);
          pdf.text(`• ${spec[0]}`, margin + 8, yOffset);
          pdf.text(spec[1], margin + 80, yOffset);
        });
        
        yPos += 55;

        // Model Performance Metrics in a professional table
        pdf.setTextColor(30, 41, 59);
        pdf.setFontSize(13);
        pdf.text('MODEL PERFORMANCE METRICS', margin, yPos);
        yPos += 10;
        
        // Table header
        pdf.setFillColor(241, 245, 249);
        pdf.setDrawColor(203, 213, 225);
        pdf.rect(margin, yPos, contentWidth, 8, 'FD');
        
        pdf.setFontSize(9);
        pdf.setTextColor(30, 58, 138);
        pdf.text('METRIC', margin + 5, yPos + 5);
        pdf.text('VALUE', margin + contentWidth/2, yPos + 5);
        
        yPos += 8;
        
        // Table rows
        const metrics = [
          ['Model Name', 'Fast Precision EfficientNet-B1'],
          ['Overall Accuracy', '91.4%'],
          ['Precision Score', '87.9%'],
          ['Recall Score', '86.0%'],
          ['F1-Score', '86.9%'],
          ['Training Dataset', '10,000+ labeled insurance images']
        ];
        
        metrics.forEach((metric, index) => {
          // Alternate row colors
          if (index % 2 === 0) {
            pdf.setFillColor(248, 250, 252);
            pdf.rect(margin, yPos, contentWidth, 6, 'F');
          }
          
          pdf.setFontSize(9);
          pdf.setTextColor(71, 85, 105);
          pdf.text(metric[0], margin + 5, yPos + 4);
          pdf.text(metric[1], margin + contentWidth/2, yPos + 4);
          
          // Row border
          pdf.setDrawColor(229, 231, 235);
          pdf.line(margin, yPos + 6, margin + contentWidth, yPos + 6);
          
          yPos += 6;
        });
        
        yPos += 10;

        // Disclaimer Section with proper styling
        pdf.setFillColor(254, 252, 232);
        pdf.setDrawColor(234, 179, 8);
        pdf.rect(margin, yPos, contentWidth, 25, 'FD');
        
        pdf.setTextColor(133, 77, 14);
        pdf.setFontSize(10);
        pdf.text('⚠️  IMPORTANT DISCLAIMER', margin + 8, yPos + 8);
        
        pdf.setFontSize(8);
        pdf.setTextColor(92, 92, 92);
        const disclaimerText = [
          'This report is generated by AI and should be used as a supplementary tool for fraud detection.',
          'Final decisions must involve human review and additional verification processes.',
          'For questions or concerns, contact your claims department supervisor immediately.'
        ];
        
        disclaimerText.forEach((line, index) => {
          pdf.text(line, margin + 8, yPos + 15 + (index * 4));
        });

        // Professional Footer
        const footerY = pageHeight - 20;
        pdf.setFillColor(30, 58, 138);
        pdf.rect(0, footerY, pageWidth, 20, 'F');
        
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(10);
        pdf.text('FraudGuard AI - Advanced Insurance Fraud Detection System', margin, footerY + 8);
        
        pdf.setFontSize(8);
        pdf.text('Confidential and Proprietary - For Internal Use Only', margin, footerY + 15);
        
        // Page number (if needed for multi-page reports)
        const pageText = 'Page 1 of 1';
        const pageTextWidth = pdf.getTextWidth(pageText);
        pdf.text(pageText, pageWidth - margin - pageTextWidth, footerY + 8);

        // Save the PDF with better filename
        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');
        const fileName = `FraudGuard-Analysis-${selectedFile?.name?.split('.')[0] || 'Report'}-${timestamp}.pdf`;
        pdf.save(fileName);
        resolve(fileName);
      }
    } catch (error) {
      reject(error);
    }
  });
};